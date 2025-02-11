from z3 import *
import pandas as pd
import numpy as np
import sys
from util import (
    transfer_to_features,
)
import multiprocessing as mp

from itertools import product
import random

TIMEOUT = 1
LEAST_ANSWERS = 1
OVERHEAD = 0.2


def call_smt(
    mode,  # ["single", "chunks"]
    df_stat,
    considered_important_feature_list,
    considered_important_feature_indices_list,
    perturb_src,
    asset_common_dir,
    tolerance,
):

    num_added_pkts_list = []
    index = int(df_stat.iloc[0]["index"])
    adversary_flow_stat = df_stat.iloc[0]
    benign_flow_stat = df_stat.iloc[1]
    num_adversary_pkts = int(np.round(adversary_flow_stat["l_num_pkts"]))
    num_benign_pkts = int(np.round(benign_flow_stat["l_num_pkts"]))
    if "l_num_pkts" in considered_important_feature_list:
        num_added_pkts = num_adversary_pkts - num_benign_pkts
        # num_added_pkts_list = [num_added_pkts]
        num_added_pkts_list = [0]

    else:
        # num_added_pkts_list = [i for i in range(20)]
        num_added_pkts_list = [0]

    if len(num_added_pkts_list) == 0:
        print("Failed to calculate num_added_pkts")

    num_splitted_pkts_percent_list = [0]  # [0.1 * i for i in range(0, 2)]

    orig_df = pd.read_csv(os.path.join(asset_common_dir, f"pkt_dir/{index}.csv"))
    # orig_df = orig_df.drop("Unnamed: 0", axis=1)

    tmp_df_results = []
    if mode == "single":
        input_list = [
            (
                df_stat,
                orig_df,
                considered_important_feature_list,
                num_added_pkts,
                num_splitted_pkts_percent,
                perturb_src,
                tolerance,
            )
            for num_added_pkts, num_splitted_pkts_percent in product(
                num_added_pkts_list, num_splitted_pkts_percent_list
            )
        ]
        tmp = smt_group_per_iter(input_list, try_smt_weak)
        for success, df_modified in tmp:
            if success:
                df_modified["time"] = df_modified["iats"].cumsum()
                tmp_df_results.append(df_modified)

    elif mode == "chunks":
        tmp_all_chunks_df_results = []
        # Each chunk is 40
        chunk_size = 5
        num_iter = int(np.ceil(len(orig_df) / chunk_size))
        if num_iter > 20:
            num_iter = 20
            chunk_size = int(np.ceil(len(orig_df) / num_iter))

        df_stat_updated = df_stat.copy()
        df_stat_updated.loc[0, "l_num_bytes"] /= num_iter
        df_stat_updated.loc[0, "l_num_pkts"] /= num_iter
        considered_important_feature_list = [
            x for x in considered_important_feature_list if "l_num_unique" not in x
        ]
        considered_important_feature_list = [
            x for x in considered_important_feature_list if "t_burst_count" not in x
        ]
        for idx in range(num_iter):
            df_this_chunk = orig_df[
                idx * chunk_size : (idx + 1) * chunk_size
            ].reset_index(drop=True)
            
            df_stat_updated.loc[0, "l_num_pkts"] = len(df_this_chunk)
            df_stat_updated.loc[1, "l_num_pkts"] = len(df_this_chunk)
            
            tmp_chunk_df_results = [df_this_chunk]
            # We need to make sure the chunk can be at least in length 1
            # if idx != num_iter - 1:
            #     num_added_pkts_list = [0]
            num_added_pkts_list = [0]
            if len(df_this_chunk) > 1:
                input_list = [
                    (
                        df_stat_updated,
                        df_this_chunk,
                        considered_important_feature_list,
                        num_added_pkts,
                        num_splitted_pkts_percent,
                        perturb_src,
                        tolerance,
                    )
                    for num_added_pkts, num_splitted_pkts_percent in product(
                        num_added_pkts_list, num_splitted_pkts_percent_list
                    )
                ]

                tmp = smt_group_per_iter(input_list, try_smt_weak)
                for success, df_chunk_modified in tmp:
                    if success:
                        tmp_chunk_df_results.append(df_chunk_modified)

            tmp_all_chunks_df_results.append(tmp_chunk_df_results)
        num_soltutions_per_chunk = [len(x) for x in tmp_all_chunks_df_results]
        picked_chunk_idx_set = []
        for _ in range(10):
            picked_chunk_idx = [np.random.randint(x) for x in num_soltutions_per_chunk]
            if all(element == 0 for element in picked_chunk_idx):
                continue
            picked_chunk_idx_set.append(picked_chunk_idx)
        for picked_chunk_idx in picked_chunk_idx_set:
            df_modified = pd.concat(
                [
                    tmp_all_chunks_df_results[i][picked_chunk_idx[i]]
                    for i in range(len(picked_chunk_idx))
                ]
            ).reset_index(drop=True)
            df_modified["time"] = df_modified["iats"].cumsum()
            tmp_df_results.append(df_modified)
    else:
        print("Invalid capability")
        exit(1)

    results = []
    for tmp_df in tmp_df_results:
        try:
            results.append(
                (
                    True,
                    transfer_to_features(tmp_df),
                    tmp_df,
                    considered_important_feature_list,
                )
            )
        except:
            results.append((False, None, None, None))
    return results


def smt_group_per_iter(input_list, smt_func):

    # result_queue = mp.Queue()
    manager = mp.Manager()
    shared_list = manager.list()
    result = []

    for i in range(len(input_list)):
        input_list[i] = input_list[i] + (shared_list,)

    process_list = []
    for i in input_list:
        process = mp.Process(target=smt_func, args=i)
        process.start()
        process_list.append(process)

    mp.connection.wait([p.sentinel for p in process_list], timeout=TIMEOUT)

    for p in process_list:
        if p.is_alive():
            p.terminate()
            p.join()

    for x in shared_list:

        result.append(x)

    return result


def try_smt_weak(
    df_stat,
    df_flow,
    considered_important_feature_list,
    num_added_pkts,
    num_splitted_pkts_percent,
    perturb_src,
    tolerance,
    tmp_queue,
):
    index = int(df_stat.iloc[0]["index"])
    upper_bound = 1 + tolerance
    lower_bound = 1 - tolerance

    # 1: original traffic
    # 0: adv traffic
    adversary_flow_stat = df_stat.iloc[0]
    benign_flow_stat = df_stat.iloc[1]
    num_adversary_pkts = int(np.round(adversary_flow_stat["l_num_pkts"]))
    num_benign_pkts = int(np.round(benign_flow_stat["l_num_pkts"]))
    df_benign_flow = df_flow

    num_existing_pkts = num_benign_pkts
    # num_added_pkts = num_adversary_pkts - num_benign_pkts
    num_added_pkts = num_added_pkts
    # Calculate interarrival for pkts and record if it's fwd or bwd
    orig_interarrival_list = df_benign_flow["iats"]

    # print(sum(orig_interarrival_list[1:]))

    orig_pkt_len_list = list(df_benign_flow["length"])

    s = Solver()
    # Define variables to represent delay of the all the pkts.
    X_existing_flow_delta_ts = [Real("dxts%s" % i) for i in range(num_existing_pkts)]
    X_existing_flow_delta_pkt_len = [
        Real("dxpl%s" % i) for i in range(num_existing_pkts)
    ]
    X_pkt_len = []
    for i in range(num_existing_pkts):
        s.add(X_existing_flow_delta_ts[i] >= 0)
        s.add(X_existing_flow_delta_pkt_len[i] == 0)
        s.add(orig_pkt_len_list[i] + X_existing_flow_delta_pkt_len[i] <= 1500)
        X_pkt_len.append(orig_pkt_len_list[i] + X_existing_flow_delta_pkt_len[i])

    # Define variables to represnet appended dumy pkts
    X_added_flow_delta_ts = [Real("axts%s" % i) for i in range(num_added_pkts)]
    X_added_flow_pkt_len = [Real("axpl%s" % i) for i in range(num_added_pkts)]

    for val in X_added_flow_pkt_len:
        X_pkt_len.append(val)

    for i in range(num_added_pkts):
        s.add(X_added_flow_pkt_len[i] >= 306)
        s.add(X_added_flow_pkt_len[i] <= 1500)
        s.add(X_added_flow_delta_ts[i] >= 0)

    interarrivals = [
        orig_interarrival_list[i] + X_existing_flow_delta_ts[i]
        for i in range(num_existing_pkts)
    ]
    for i in range(num_added_pkts):
        interarrivals.append(X_added_flow_delta_ts[i])
    s.add(Sum(interarrivals[1:]) <= (1 + OVERHEAD) * sum(orig_interarrival_list))

    # Define the constraints

    # l_max
    if "l_max" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    X_pkt_len[i] <= upper_bound * adversary_flow_stat["l_max"]
                    for i in range(len(X_pkt_len))
                ]
            )
        )
        s.add(
            z3.Or(
                [
                    z3.And(
                        [
                            X_pkt_len[i] >= lower_bound * adversary_flow_stat["l_max"],
                            X_pkt_len[i] <= upper_bound * adversary_flow_stat["l_max"],
                        ]
                    )
                    for i in range(len(X_pkt_len))
                ]
            )
        )

    if "l_min" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    X_pkt_len[i] >= lower_bound * adversary_flow_stat["l_min"]
                    for i in range(len(X_pkt_len))
                ]
            )
        )
        s.add(
            z3.Or(
                [
                    z3.And(
                        [
                            X_pkt_len[i] >= lower_bound * adversary_flow_stat["l_min"],
                            X_pkt_len[i] <= upper_bound * adversary_flow_stat["l_min"],
                        ]
                    )
                    for i in range(len(X_pkt_len))
                ]
            )
        )

    if "l_mean" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    Sum(X_pkt_len)
                    >= lower_bound * adversary_flow_stat["l_mean"] * len(X_pkt_len),
                    Sum(X_pkt_len)
                    <= upper_bound * adversary_flow_stat["l_mean"] * len(X_pkt_len),
                ]
            )
        )

    if "l_std" in considered_important_feature_list:
        std_pkt_len = [(i - Sum(X_pkt_len) / len(X_pkt_len)) ** 2 for i in X_pkt_len]
        s.add(
            z3.And(
                [
                    Sum(std_pkt_len) / len(std_pkt_len)
                    >= lower_bound * adversary_flow_stat["l_std"] ** 2,
                    Sum(std_pkt_len) / len(std_pkt_len)
                    <= upper_bound * adversary_flow_stat["l_std"] ** 2,
                ]
            )
        )

    if "l_q2" in considered_important_feature_list:
        pkt_len_median = Real("pkt_len_median")
        for pkt_len in X_pkt_len:
            s.add(Or(pkt_len >= pkt_len_median, pkt_len <= pkt_len_median))
        half_size = (len(X_pkt_len) + 1) // 2
        s.add(
            Sum([If(pkt_len <= pkt_len_median, 1, 0) for pkt_len in X_pkt_len])
            >= half_size
        )
        s.add(
            Sum([If(pkt_len >= pkt_len_median, 1, 0) for pkt_len in X_pkt_len])
            >= half_size
        )

    if "t_min" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    interarrivals[i] >= lower_bound * adversary_flow_stat["t_min"]
                    for i in range(len(interarrivals))
                ]
            )
        )
        s.add(
            z3.Or(
                [
                    z3.And(
                        [
                            interarrivals[i]
                            >= lower_bound * adversary_flow_stat["t_min"],
                            interarrivals[i]
                            <= upper_bound * adversary_flow_stat["t_min"],
                        ]
                    )
                    for i in range(len(interarrivals))
                ]
            )
        )

    if "t_max" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    interarrivals[i] <= upper_bound * adversary_flow_stat["t_max"]
                    for i in range(len(interarrivals))
                ]
            )
        )
        s.add(
            z3.Or(
                [
                    z3.And(
                        [
                            interarrivals[i]
                            >= lower_bound * adversary_flow_stat["t_max"],
                            interarrivals[i]
                            <= upper_bound * adversary_flow_stat["t_max"],
                        ]
                    )
                    for i in range(len(interarrivals))
                ]
            )
        )

    if "t_mean" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    Sum(interarrivals)
                    >= lower_bound * adversary_flow_stat["t_mean"] * len(interarrivals),
                    Sum(interarrivals)
                    <= upper_bound * adversary_flow_stat["t_mean"] * len(interarrivals),
                ]
            )
        )

    if "t_std" in considered_important_feature_list:
        std_interarrivals = [
            (interarrivals[i] - Sum(interarrivals) / len(interarrivals)) ** 2
            for i in range(len(interarrivals))
        ]
        s.add(
            z3.And(
                [
                    Sum(std_interarrivals) / len(std_interarrivals)
                    >= lower_bound * adversary_flow_stat["t_std"] ** 2,
                    Sum(std_interarrivals) / len(std_interarrivals)
                    <= upper_bound * adversary_flow_stat["t_std"] ** 2,
                ]
            )
        )

    if "t_q2" in considered_important_feature_list:
        interarrivals_median = Real("interarrivals_median")
        for interarrival in interarrivals:
            s.add(
                Or(
                    interarrival >= interarrivals_median,
                    interarrival <= interarrivals_median,
                )
            )
        half_size = (len(interarrivals) + 1) // 2
        s.add(
            Sum(
                [
                    If(interarrival <= interarrivals_median, 1, 0)
                    for interarrival in interarrivals
                ]
            )
            >= half_size
        )
        s.add(
            Sum(
                [
                    If(interarrival >= interarrivals_median, 1, 0)
                    for interarrival in interarrivals
                ]
            )
            >= half_size
        )

    if "l_num_bytes" in considered_important_feature_list:
        s.add(
            Sum(X_pkt_len) >= lower_bound * adversary_flow_stat["l_num_bytes"],
            Sum(X_pkt_len) <= upper_bound * adversary_flow_stat["l_num_bytes"],
        )

    if "l_num_unique" in considered_important_feature_list:
        is_unique = [Bool(f"unique{i}") for i in range(len(X_pkt_len))]
        for i in range(len(X_pkt_len)):
            # An element is unique if there is no other element with the same value
            other_elements = [X_pkt_len[j] for j in range(len(X_pkt_len)) if j != i]
            s.add(
                is_unique[i]
                == And(
                    [
                        Or(
                            X_pkt_len[i] > upper_bound * o,
                            X_pkt_len[i] < lower_bound * o,
                        )
                        for o in other_elements
                    ]
                )
            )
        num_unique = Sum([If(is_unique[i], 1, 0) for i in range(len(X_pkt_len))])

        s.add(
            num_unique >= lower_bound * adversary_flow_stat["l_num_unique"],
            num_unique <= upper_bound * adversary_flow_stat["l_num_unique"],
        )

    if "t_burst_count" in considered_important_feature_list:
        s.add(
            Sum([If(interarrival >= 30, 1, 0) for interarrival in interarrivals])
            >= lower_bound * int(np.round(adversary_flow_stat["t_burst_count"]))
        )
        s.add(
            Sum([If(interarrival >= 30, 1, 0) for interarrival in interarrivals])
            <= upper_bound * int(np.round(adversary_flow_stat["t_burst_count"]))
        )

    success = False
    # Check for a solution
    if s.check() == sat:
        solution = s.model()
        modified = {
            "length": [],
            "iats": [],
        }

        ts_solution = []
        for i in range(num_existing_pkts):
            delay = solution[Real("dxts%s" % i)].as_fraction()
            if i == 0:
                ts_solution.append(
                    orig_interarrival_list[i]
                    + float(delay.numerator) / float(delay.denominator)
                )
            else:
                ts_solution.append(
                    ts_solution[i - 1]
                    + orig_interarrival_list[i]
                    + float(delay.numerator) / float(delay.denominator)
                )
        pkt_len_solution = []
        for i in range(num_existing_pkts):
            delta_pkt_len = solution[Real("dxpl%s" % i)].as_fraction()
            pkt_len_solution.append(
                df_benign_flow["length"][i]
                + np.round(
                    float(delta_pkt_len.numerator) / float(delta_pkt_len.denominator)
                )
            )

        for i in range(num_added_pkts):
            delay = solution[Real("axts%s" % i)].as_fraction()
            ts_solution.append(
                ts_solution[-1] + float(delay.numerator) / float(delay.denominator)
            )
            delta_pkt_len = solution[Real("axpl%s" % i)].as_fraction()
            pkt_len_solution.append(
                np.round(
                    float(delta_pkt_len.numerator) / float(delta_pkt_len.denominator)
                )
            )

        interarrival_solution = []
        for i in range(num_existing_pkts + num_added_pkts):
            if i < num_existing_pkts:
                delay = solution[Real("dxts%s" % i)].as_fraction()
            else:
                delay = solution[Real("axts%s" % (i - num_existing_pkts))].as_fraction()
            if i == 0:
                interarrival_solution.append(
                    orig_interarrival_list[i]
                    + float(delay.numerator) / float(delay.denominator)
                )
            else:
                interarrival_solution.append(ts_solution[i] - ts_solution[i - 1])

        for i in range(num_existing_pkts + num_added_pkts):
            modified["length"].append(np.round(pkt_len_solution[i]))
            modified["iats"].append(interarrival_solution[i])

        success = True
        df_modified = pd.DataFrame(modified)
    else:
        success = False
        df_modified = None
    tmp_queue.append((success, df_modified))
    return success, df_modified
