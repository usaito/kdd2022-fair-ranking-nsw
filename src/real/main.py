from logging import getLogger
from pathlib import Path
import pickle
from time import time

from func import compute_pi_expo_fair
from func import compute_pi_max
from func import compute_pi_nsw
from func import compute_pi_unif
from func import evaluate_pi
from func import exam_func
from func import preprocess_data
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from pandas import DataFrame


logger = getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # configurations
    K_list = cfg.setting.K_list
    n_doc = cfg.setting.n_doc
    alpha_list = cfg.setting.alpha_list
    pol_list = ["max", "expo-fair", "unif"]
    pol_list += [f"nsw_{alpha}" for alpha in alpha_list]
    result_df = DataFrame(
        columns=[
            "policy",
            "seed",
            "K",
            "user_util",
            "mean_max_envy",
            "pct_better_off",
            "pct_worse_off",
        ]
    )

    # log path
    log_path = Path("./varying_K")
    log_path.mkdir(exist_ok=True, parents=True)

    elapsed_prev = 0.0
    for k in K_list:
        v = exam_func(n_doc, k, cfg.setting.exam_func)
        result_df_box = DataFrame(
            columns=["policy", "item_util", "max_envy", "imp_in_item_util"]
        )
        for s in np.arange(cfg.setting.n_seeds):
            rel_mat_true, rel_mat_obs, info = preprocess_data(
                dataset=cfg.setting.dataset,
                path=Path().cwd().parents[2] / "data",
                n_doc=n_doc,
                test_size=cfg.setting.test_size,
                random_state=s,
            )

            # uniform
            pi_unif = compute_pi_unif(rel_mat=rel_mat_obs, v=v)
            user_util_unif, item_utils_unif, max_envies_unif, _ = evaluate_pi(
                pi=pi_unif,
                rel_mat=rel_mat_true,
                v=v,
            )

            # max
            pi_max = compute_pi_max(rel_mat=rel_mat_obs, v=v)
            user_util_max, item_utils_max, max_envies_max, _ = evaluate_pi(
                pi=pi_max,
                rel_mat=rel_mat_true,
                v=v,
            )
            result_max = DataFrame(
                data={
                    "policy": ["Max"] * rel_mat_true.shape[1],
                    "item_util": item_utils_max,
                    "max_envy": max_envies_max,
                    "imp_in_item_util": item_utils_max / item_utils_unif,
                }
            )
            result_df_box = pd.concat([result_df_box, result_max])

            # exposure-based fair
            pi_expo_fair = compute_pi_expo_fair(rel_mat=rel_mat_obs, v=v)
            user_util_expo, item_utils_expo, max_envies_expo, _ = evaluate_pi(
                pi=pi_expo_fair,
                rel_mat=rel_mat_true,
                v=v,
            )
            result_expo_fair = DataFrame(
                data={
                    "policy": ["Expo-Fair"] * rel_mat_true.shape[1],
                    "item_util": item_utils_expo,
                    "max_envy": max_envies_expo,
                    "imp_in_item_util": item_utils_expo / item_utils_unif,
                }
            )
            result_df_box = pd.concat([result_df_box, result_expo_fair])

            # nash social welfare
            user_util_nsw_list = []
            mean_max_envy_nsw_list = []
            pct_item_util_better_off_nsw_list = []
            pct_item_util_worse_off_nsw_list = []
            for alpha_ in alpha_list:
                pi_nsw = compute_pi_nsw(rel_mat=rel_mat_obs, v=v, alpha=alpha_)
                user_util_nsw, item_utils_nsw, max_envies_nsw, _ = evaluate_pi(
                    pi=pi_nsw,
                    rel_mat=rel_mat_true,
                    v=v,
                )
                imp_item_util_ = item_utils_nsw / item_utils_unif
                result_nsw = DataFrame(
                    data={
                        "policy": [f"NSW ({alpha_})"] * rel_mat_true.shape[1],
                        "item_util": item_utils_nsw,
                        "max_envy": max_envies_nsw,
                        "imp_in_item_util": imp_item_util_,
                    }
                )
                result_df_box = pd.concat([result_df_box, result_nsw])
                user_util_nsw_list.append(user_util_nsw)
                mean_max_envy_nsw_list.append(max_envies_nsw.mean())
                pct_item_util_better_off_nsw_list.append(
                    100 * (imp_item_util_ > 1.10).mean()
                )
                pct_item_util_worse_off_nsw_list.append(
                    100 * (imp_item_util_ < 0.90).mean()
                )

            # aggregate results
            user_util_list = [
                user_util_max,
                user_util_expo,
                user_util_unif,
            ]
            user_util_list += user_util_nsw_list
            mean_max_envy_list = [
                max_envies_max.mean(),
                max_envies_expo.mean(),
                0.0,
            ]
            mean_max_envy_list += mean_max_envy_nsw_list
            pct_item_util_better_off_list = [
                ((item_utils_max / item_utils_unif) > 1.10).mean(),
                ((item_utils_expo / item_utils_unif) > 1.10).mean(),
                ((item_utils_unif / item_utils_unif) > 1.10).mean(),
            ]
            pct_item_util_better_off_list += pct_item_util_better_off_nsw_list
            pct_item_util_worse_off_list = [
                ((item_utils_max / item_utils_unif) < 0.90).mean(),
                ((item_utils_expo / item_utils_unif) < 0.90).mean(),
                ((item_utils_unif / item_utils_unif) < 0.90).mean(),
            ]
            pct_item_util_worse_off_list += pct_item_util_worse_off_nsw_list

            result_df_ = DataFrame(
                data={
                    "policy": pol_list,
                    "seed": [s] * len(pol_list),
                    "K": [k] * len(pol_list),
                    "user_util": user_util_list,
                    "mean_max_envy": mean_max_envy_list,
                    "pct_better_off": pct_item_util_better_off_list,
                    "pct_worse_off": pct_item_util_worse_off_list,
                },
            )
            result_df = pd.concat([result_df, result_df_])
        elapsed = np.round((time() - start_time) / 60, 2)
        diff = np.round(elapsed - elapsed_prev, 2)
        logger.info(f"k={k}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

        result_df_box.reset_index(inplace=True)
        result_df_box.to_csv(log_path / f"result_df_box_k={k}.csv")
        with open(log_path / f"info_k={k}.pkl", "wb") as tf:
            pickle.dump(info, tf)

    result_df.reset_index(inplace=True)
    result_df.to_csv(log_path / "result_df.csv")


if __name__ == "__main__":
    main()
