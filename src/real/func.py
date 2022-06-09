from pathlib import Path

import cvxpy as cvx
import numpy as np
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from xclib.data import data_utils


def preprocess_data(
    dataset: str,
    path: Path = Path("./data"),
    n_doc: int = 100,
    classifier: ClassifierMixin = LogisticRegression(
        C=100,
        max_iter=1000,
        random_state=12345,
    ),
    lam: float = 1.0,
    test_size: float = 0.1,
    eps_plus: float = 1.0,
    eps_minus: float = 0.0,
    random_state: int = 12345,
) -> list:
    info = dict()
    random_ = check_random_state(random_state)

    # Read file with features and labels
    if dataset == "d":
        dataset_ = "delicious"
    elif dataset == "e":
        dataset_ = "eurlex"
    elif dataset == "w":
        dataset_ = "wiki"
    features, tabels, num_samples, num_features, num_labels = data_utils.read_data(
        path / f"{dataset_}.txt"
    )
    info["num_all_data"] = num_samples
    info["num_features"] = num_features
    info["num_labels"] = num_labels
    info["n_doc"] = n_doc

    # BoW Feature
    X = features.toarray()
    # Multilabel Table
    T = tabels.toarray()

    # sample labels via eps-greedy
    n_points_per_label = T.sum(0)
    top_labels = rankdata(-n_points_per_label, method="ordinal") <= n_doc
    sampling_probs = lam * top_labels + (1 - lam) * n_doc / num_labels
    sampling_probs /= sampling_probs.sum()
    sampled_labels = random_.choice(
        np.arange(num_labels), size=n_doc, p=sampling_probs, replace=False
    )
    T = T[:, sampled_labels]
    info["lam"] = lam

    # minimum num of relevant labels per data
    if dataset == "d":
        n_rel_labels = 10
    elif dataset == "e":
        n_rel_labels = 2
    elif dataset == "w":
        n_rel_labels = 5
    n_rel_labels = np.maximum(n_rel_labels * lam, 1)
    n_labels_per_point = T.sum(1)
    T = T[n_labels_per_point >= n_rel_labels, :]
    X = X[n_labels_per_point >= n_rel_labels, :]

    # train-test split
    X_tr, X_te, rel_mat_tr, rel_mat_te = train_test_split(
        X,
        T,
        test_size=test_size,
        random_state=random_state,
    )
    rel_mat_obs = rel_mat_tr[:, (rel_mat_tr.sum(0) != 0) & (rel_mat_te.sum(0) != 0)]
    rel_mat_te = rel_mat_te[:, (rel_mat_tr.sum(0) != 0) & (rel_mat_te.sum(0) != 0)]
    info["num_train"] = X_tr.shape[0]
    info["num_test"] = X_te.shape[0]

    # add noise to train data
    rel_mat_tr_prob = eps_plus * rel_mat_obs + eps_minus * (1 - rel_mat_obs)
    rel_mat_tr_obs = random_.binomial(n=1, p=rel_mat_tr_prob)

    # relevance prediction
    n_components = np.minimum(200, rel_mat_obs.shape[0])
    pipe = Pipeline(
        [
            ("pca", PCA(n_components=n_components)),
            ("scaler", StandardScaler()),
            (
                "clf",
                MultiOutputClassifier(
                    classifier,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, rel_mat_tr_obs)
    rel_mat_te_pred = np.concatenate(
        [_[:, 1][:, np.newaxis] for _ in pipe.predict_proba(X_te)], 1
    )
    info["explained_variance_ratio"] = pipe["pca"].explained_variance_ratio_.sum()
    info["log_loss"] = log_loss(rel_mat_te.flatten(), rel_mat_te_pred.flatten())
    info["auc"] = roc_auc_score(rel_mat_te.flatten(), rel_mat_te_pred.flatten())

    return rel_mat_te, rel_mat_te_pred, info


def exam_func(n_doc: int, K: int, shape: str = "inv") -> np.ndarray:
    assert shape in ["inv", "exp", "log"]
    if shape == "inv":
        v = np.ones(K) / np.arange(1, K + 1)
    elif shape == "exp":
        v = 1.0 / np.exp(np.arange(K))

    return v[:, np.newaxis]


def evaluate_pi(pi: np.ndarray, rel_mat: np.ndarray, v: np.ndarray):
    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils = click_mat.sum(0) / n_query
    nsw = np.power(item_utils.prod(), 1 / n_doc)

    max_envies = np.zeros(n_doc)
    for i in range(n_doc):
        u_d_swap = (expo_mat * rel_mat[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies[i] = d_envies.max() / n_query

    return user_util, item_utils, max_envies, nsw


def compute_pi_max(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    pi = np.zeros((n_query, n_doc, K))
    for k in np.arange(K):
        pi_at_k = np.zeros_like(rel_mat)
        pi_at_k[rankdata(-rel_mat, axis=1, method="ordinal") == k + 1] = 1
        pi[:, :, k] = pi_at_k

    return pi


def compute_pi_expo_fair(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0)[:, np.newaxis]
    am_expo = v.sum() * n_query * am_rel / rel_mat.sum()

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        pi_d = pi[:, K * d : K * (d + 1)]
        obj += rel_mat[:, d] @ pi_d @ v
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
        # amortized exposure
        constraints += [query_basis.T @ pi_d @ v <= am_expo[d]]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_nsw(
    rel_mat: np.ndarray,
    v: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0) ** alpha

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        obj += am_rel[d] * cvx.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ v)
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_unif(rel_mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    return np.ones((n_query, n_doc, K)) / n_doc
