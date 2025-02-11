from z3 import *
import pandas as pd
import numpy as np
import sys
from util import (
    calculate_active_idle_len,
    calculate_active_idle_index,
    transfer_to_features,
)
import multiprocessing as mp

from itertools import product
import random

TIMEOUT = 1
LEAST_ANSWERS = 1
OVERHEAD = 0.2
PKT_PAYLOAD_APPEND_LIMIT = 0.2

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
    five_tuples = ["srcip", "dstip", "srcport", "dstport", "proto"]
    index = int(df_stat.iloc[0]["index"])

    # 1: original traffic
    # 0: adv traffic
    adversary_bi_flow_stat = df_stat.iloc[0]
    benign_bi_flow_stat = df_stat.iloc[1]

    if (
        "total_fiat" in considered_important_feature_list
        and "mean_fiat" in considered_important_feature_list
    ):
        num_adversary_forward_pkts = int(
            np.round(
                adversary_bi_flow_stat["total_fiat"]
                / adversary_bi_flow_stat["mean_fiat"]
            )
            + 1
        )
        num_benign_forward_pkts = int(
            np.round(
                benign_bi_flow_stat["total_fiat"] / benign_bi_flow_stat["mean_fiat"]
            )
            + 1
        )
        num_added_pkts = num_adversary_forward_pkts - num_benign_forward_pkts
        if num_added_pkts < 0:
            num_added_pkts = 0
        num_added_pkts_list = [num_added_pkts]

    elif (
        "duration" in considered_important_feature_list
        and "flowPktsPerSecond" in considered_important_feature_list
    ):
        num_adversary_total_pkts = int(
            np.round(
                adversary_bi_flow_stat["duration"]
                / 10**6
                * adversary_bi_flow_stat["flowPktsPerSecond"]
            )
        )
        num_benign_forward_pkts = int(
            np.round(
                benign_bi_flow_stat["total_fiat"] / benign_bi_flow_stat["mean_fiat"]
            )
            + 1
        )
        num_benign_backward_pkts = int(
            np.round(
                benign_bi_flow_stat["total_biat"] / benign_bi_flow_stat["mean_biat"]
            )
            + 1
        )
        num_added_pkts = (
            num_adversary_total_pkts
            - num_benign_backward_pkts
            - num_benign_forward_pkts
        )
        if num_added_pkts < 0:
            num_added_pkts = 0
        num_added_pkts_list = [num_added_pkts]

    else:
        num_added_pkts_list = num_added_pkts_list = [i for i in range(20)]

    num_splitted_pkts_percent_list = [0]  # [0.1 * i for i in range(0, 2)]

    if len(num_added_pkts_list) == 0:
        print("Failed to calculate num_added_pkts")

    orig_df = pd.read_csv(os.path.join(asset_common_dir, f"pkt_dir/{index}.csv"))
    orig_df = orig_df.drop("Unnamed: 0", axis=1)
    # calculate iat
    orig_df["iat"] = orig_df["time"].diff()
    orig_df["iat"] = orig_df["iat"].fillna(0)
    orig_df = orig_df.drop(["time"], axis=1)
    fwd_traffic_five_tuple = list(orig_df.iloc[0][five_tuples])

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
                fwd_traffic_five_tuple,
                tolerance,
                asset_common_dir,
            )
            for num_added_pkts, num_splitted_pkts_percent in product(
                num_added_pkts_list, num_splitted_pkts_percent_list
            )
        ]
        tmp = smt_group_per_iter(input_list, try_smt_strong)
        for success, df_modified in tmp:
            if success:
                df_modified["time"] = df_modified["iat"].cumsum()
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
        df_stat_updated.loc[0, "total_fiat"] /= num_iter
        df_stat_updated.loc[0, "total_biat"] /= num_iter
        df_stat_updated.loc[0, "duration"] /= num_iter
        considered_important_feature_list = [x for x in considered_important_feature_list if "active" not in x]
        considered_important_feature_list = [x for x in considered_important_feature_list if "idle" not in x]
        for idx in range(num_iter):
            df_this_chunk = orig_df[
                idx * chunk_size : (idx + 1) * chunk_size
            ].reset_index(drop=True)

            tmp_chunk_df_results = [df_this_chunk]
            # We need to make sure the chunk can be at least in length 1
            # if idx != num_iter - 1:
            #     num_added_pkts_list = [0]
            num_added_pkts_list = [0, 1]
            if len(df_this_chunk) > 1:
                input_list = [
                    (
                        df_stat_updated,
                        df_this_chunk,
                        considered_important_feature_list,
                        num_added_pkts,
                        num_splitted_pkts_percent,
                        perturb_src,
                        fwd_traffic_five_tuple,
                        tolerance,
                        asset_common_dir,
                    )
                    for num_added_pkts, num_splitted_pkts_percent in product(
                        num_added_pkts_list, num_splitted_pkts_percent_list
                    )
                ]

                tmp = smt_group_per_iter(input_list, try_smt_strong)
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
            df_modified["time"] = df_modified["iat"].cumsum()
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
                    transfer_to_features(tmp_df, 1e5),
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


def try_smt_strong(
    df_stat,
    df_flow,
    considered_important_feature_list,
    num_added_pkts,
    num_splitted_pkts_percent,
    perturb_src,
    fwd_traffic_five_tuple,
    tolerance,
    asset_common_dir,
    tmp_queue,
):
    five_tuples = ["srcip", "dstip", "srcport", "dstport", "proto"]
    index = int(df_stat.iloc[0]["index"])
    upper_bound = 1 + tolerance
    lower_bound = 1 - tolerance

    # 1: original traffic
    # 0: adv traffic
    adversary_bi_flow_stat = df_stat.iloc[0]
    benign_bi_flow_stat = df_stat.iloc[1]

    active_threshold = 1e5
    active_len, idle_len = calculate_active_idle_len(index, active_threshold, asset_common_dir)

    df_benign_bi_flow = df_flow

    bwd_traffic_five_tuple = [
        fwd_traffic_five_tuple[1],
        fwd_traffic_five_tuple[0],
        fwd_traffic_five_tuple[3],
        fwd_traffic_five_tuple[2],
        fwd_traffic_five_tuple[4],
    ]

    num_existing_pkts = len(df_benign_bi_flow)

    # Calculate interarrival for pkts and record if it's fwd or bwd
    # If fwd, record it as 0. If bwd, record it as 1
    orig_interarrival_list = df_benign_bi_flow["iat"]

    fwd_bwd_indicator = []
    orig_fwd_pkt_len_list = []
    for i in range(len(df_benign_bi_flow)):
        tmp = list(df_benign_bi_flow.iloc[i][five_tuples])
        if tmp == fwd_traffic_five_tuple:
            fwd_bwd_indicator.append(0)
            orig_fwd_pkt_len_list.append(df_benign_bi_flow.iloc[i]["pkt_len"])
        elif tmp == bwd_traffic_five_tuple:
            fwd_bwd_indicator.append(1)
        else:
            print("ERROR")
            exit(1)

    s = Solver()

    # Define variables to represent delay of the all the pkts. The delay for the backward pkts should be 0
    X_existing_flow_delta_ts = [Real("dxts%s" % i) for i in range(num_existing_pkts)]
    X_existing_flow_delta_pkt_len = [
        Real("dxpl%s" % i) for i in range(num_existing_pkts)
    ]

    if perturb_src:
        pkt_appendable_idx = [i for i in range(num_existing_pkts) if fwd_bwd_indicator[i] == 0]
    else:
        pkt_appendable_idx = [i for i in range(num_existing_pkts) if fwd_bwd_indicator[i] == 1]
    random.shuffle(pkt_appendable_idx)
    tmp_len = len(pkt_appendable_idx)
    pkt_appendable_idx = pkt_appendable_idx[:int(PKT_PAYLOAD_APPEND_LIMIT * tmp_len)]

    for i in range(num_existing_pkts):
        if fwd_bwd_indicator[i] == 0:
            if perturb_src:
                s.add(X_existing_flow_delta_ts[i] >= 0)
            else:
                s.add(X_existing_flow_delta_ts[i] == 0)
            if perturb_src and i in pkt_appendable_idx:
                s.add(X_existing_flow_delta_pkt_len[i] >= 0)
            else:
                s.add(X_existing_flow_delta_pkt_len[i] == 0)
        elif fwd_bwd_indicator[i] == 1:
            if perturb_src:
                s.add(X_existing_flow_delta_ts[i] == 0)
            else:
                s.add(X_existing_flow_delta_ts[i] >= 0)
            if perturb_src == False and i in pkt_appendable_idx:
                s.add(X_existing_flow_delta_pkt_len[i] >= 0)
            else:
                s.add(X_existing_flow_delta_pkt_len[i] == 0)
        else:
            print("ERROR")
            exit(1)

    # Define variables to represnet appended dumy pkts
    X_added_flow_delta_ts = [Real("axts%s" % i) for i in range(num_added_pkts)]
    X_added_flow_pkt_len = [Real("axpl%s" % i) for i in range(num_added_pkts)]

    for i in range(num_added_pkts):
        s.add(X_added_flow_delta_ts[i] >= 1)
        s.add(X_added_flow_pkt_len[i] >= 40)
        s.add(X_added_flow_pkt_len[i] <= 1500)
        if perturb_src:
            fwd_bwd_indicator.append(0)
        else:
            fwd_bwd_indicator.append(1)

    flow_abs_ts = []
    for i in range(num_existing_pkts):
        if i == 0:
            flow_abs_ts.append(orig_interarrival_list[i] + X_existing_flow_delta_ts[i])
        else:
            flow_abs_ts.append(
                flow_abs_ts[-1]
                + orig_interarrival_list[i]
                + X_existing_flow_delta_ts[i]
            )
    for i in range(num_added_pkts):
        flow_abs_ts.append(flow_abs_ts[-1] + X_added_flow_delta_ts[i])
    s.add((flow_abs_ts[-1] - flow_abs_ts[0]) <= (1 + OVERHEAD) * sum(orig_interarrival_list))

    # duration
    if "duration" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    flow_abs_ts[-1] - flow_abs_ts[0]
                    >= lower_bound * adversary_bi_flow_stat["duration"],
                    flow_abs_ts[-1] - flow_abs_ts[0]
                    <= upper_bound * adversary_bi_flow_stat["duration"],
                ]
            )
        )

    fwd_indices = [i for i, x in enumerate(fwd_bwd_indicator) if x == 0]
    if len(fwd_indices) >= 0:
        # total_fiat
        if "total_fiat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        flow_abs_ts[fwd_indices[-1]] - flow_abs_ts[fwd_indices[0]]
                        >= lower_bound * adversary_bi_flow_stat["total_fiat"],
                        flow_abs_ts[fwd_indices[-1]] - flow_abs_ts[fwd_indices[0]]
                        <= upper_bound * adversary_bi_flow_stat["total_fiat"],
                    ]
                )
            )

        # min_fiat
        if "min_fiat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        (flow_abs_ts[fwd_indices[i + 1]] - flow_abs_ts[fwd_indices[i]])
                        >= lower_bound * adversary_bi_flow_stat["min_fiat"]
                        for i in range(len(fwd_indices) - 1)
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                (
                                    flow_abs_ts[fwd_indices[i + 1]]
                                    - flow_abs_ts[fwd_indices[i]]
                                )
                                >= lower_bound * adversary_bi_flow_stat["min_fiat"],
                                (
                                    flow_abs_ts[fwd_indices[i + 1]]
                                    - flow_abs_ts[fwd_indices[i]]
                                )
                                <= upper_bound * adversary_bi_flow_stat["min_fiat"],
                            ]
                        )
                        for i in range(len(fwd_indices) - 1)
                    ]
                )
            )

        # max_fiat
        if "max_fiat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        (flow_abs_ts[fwd_indices[i + 1]] - flow_abs_ts[fwd_indices[i]])
                        <= upper_bound * adversary_bi_flow_stat["max_fiat"]
                        for i in range(len(fwd_indices) - 1)
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                (
                                    flow_abs_ts[fwd_indices[i + 1]]
                                    - flow_abs_ts[fwd_indices[i]]
                                )
                                >= lower_bound * adversary_bi_flow_stat["max_fiat"],
                                (
                                    flow_abs_ts[fwd_indices[i + 1]]
                                    - flow_abs_ts[fwd_indices[i]]
                                )
                                <= upper_bound * adversary_bi_flow_stat["max_fiat"],
                            ]
                        )
                        for i in range(len(fwd_indices) - 1)
                    ]
                )
            )

        # mean_fiat
        if "mean_fiat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        flow_abs_ts[fwd_indices[-1]] - flow_abs_ts[fwd_indices[0]]
                        >= lower_bound
                        * adversary_bi_flow_stat["mean_fiat"]
                        * (len(fwd_indices) - 1),
                        flow_abs_ts[fwd_indices[-1]] - flow_abs_ts[fwd_indices[0]]
                        <= upper_bound
                        * adversary_bi_flow_stat["mean_fiat"]
                        * (len(fwd_indices) - 1),
                    ]
                )
            )

    bwd_indices = [i for i, x in enumerate(fwd_bwd_indicator) if x == 1]
    if len(bwd_indices) >= 2:
        # total_biat
        if "total_biat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        flow_abs_ts[bwd_indices[-1]] - flow_abs_ts[bwd_indices[0]]
                        >= lower_bound * adversary_bi_flow_stat["total_biat"],
                        flow_abs_ts[bwd_indices[-1]] - flow_abs_ts[bwd_indices[0]]
                        <= upper_bound * adversary_bi_flow_stat["total_biat"],
                    ]
                )
            )

        # min_biat
        if "min_biat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        (flow_abs_ts[bwd_indices[i + 1]] - flow_abs_ts[bwd_indices[i]])
                        >= lower_bound * adversary_bi_flow_stat["min_biat"]
                        for i in range(len(bwd_indices) - 1)
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                (
                                    flow_abs_ts[bwd_indices[i + 1]]
                                    - flow_abs_ts[bwd_indices[i]]
                                )
                                >= lower_bound * adversary_bi_flow_stat["min_biat"],
                                (
                                    flow_abs_ts[bwd_indices[i + 1]]
                                    - flow_abs_ts[bwd_indices[i]]
                                )
                                <= upper_bound * adversary_bi_flow_stat["min_biat"],
                            ]
                        )
                        for i in range(len(bwd_indices) - 1)
                    ]
                )
            )

        # max_biat
        if "max_biat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        (flow_abs_ts[bwd_indices[i + 1]] - flow_abs_ts[bwd_indices[i]])
                        <= upper_bound * adversary_bi_flow_stat["max_biat"]
                        for i in range(len(bwd_indices) - 1)
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                (
                                    flow_abs_ts[bwd_indices[i + 1]]
                                    - flow_abs_ts[bwd_indices[i]]
                                )
                                >= lower_bound * adversary_bi_flow_stat["max_biat"],
                                (
                                    flow_abs_ts[bwd_indices[i + 1]]
                                    - flow_abs_ts[bwd_indices[i]]
                                )
                                <= upper_bound * adversary_bi_flow_stat["max_biat"],
                            ]
                        )
                        for i in range(len(bwd_indices) - 1)
                    ]
                )
            )

        # mean_biat
        if "mean_biat" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        flow_abs_ts[bwd_indices[-1]] - flow_abs_ts[bwd_indices[0]]
                        >= lower_bound
                        * adversary_bi_flow_stat["mean_biat"]
                        * (len(bwd_indices) - 1),
                        flow_abs_ts[bwd_indices[-1]] - flow_abs_ts[bwd_indices[0]]
                        <= upper_bound
                        * adversary_bi_flow_stat["mean_biat"]
                        * (len(bwd_indices) - 1),
                    ]
                )
            )

    interarrivals = [
        flow_abs_ts[i + 1] - flow_abs_ts[i] for i in range(len(flow_abs_ts) - 1)
    ]
    for interarrival in interarrivals:
        s.add(interarrival >= 0)

    # min_flowiat
    if "min_flowiat" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    interarrivals[i] >= lower_bound * adversary_bi_flow_stat["min_flowiat"]
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
                            >= lower_bound * adversary_bi_flow_stat["min_flowiat"],
                            interarrivals[i]
                            <= upper_bound * adversary_bi_flow_stat["min_flowiat"],
                        ]
                    )
                    for i in range(len(interarrivals))
                ]
            )
        )

    # mean_flowiat
    if "mean_flowiat" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    Sum(interarrivals) / len(interarrivals)
                    >= lower_bound * adversary_bi_flow_stat["mean_flowiat"],
                    Sum(interarrivals) / len(interarrivals)
                    <= upper_bound * adversary_bi_flow_stat["mean_flowiat"],
                ]
            )
        )

    # max_flowiat
    if "max_flowiat" in considered_important_feature_list:
        s.add(
            z3.And(
                [
                    interarrivals[i] <= upper_bound * adversary_bi_flow_stat["max_flowiat"]
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
                            >= lower_bound * adversary_bi_flow_stat["max_flowiat"],
                            interarrivals[i]
                            <= upper_bound * adversary_bi_flow_stat["max_flowiat"],
                        ]
                    )
                    for i in range(len(interarrivals))
                ]
            )
        )

    # std_flowiat
    if "std_flowiat" in considered_important_feature_list:
        std_interarrivals = [
            (i - Sum(interarrivals) / len(interarrivals)) ** 2 for i in interarrivals
        ]
        s.add(
            z3.And(
                [
                    Sum(std_interarrivals) / len(std_interarrivals)
                    >= lower_bound * adversary_bi_flow_stat["std_flowiat"] ** 2,
                    Sum(std_interarrivals) / len(std_interarrivals)
                    <= upper_bound * adversary_bi_flow_stat["std_flowiat"] ** 2,
                ]
            )
        )

    # pkt_len
    required_features = ["flowBytesPerSecond", "duration"]
    if set(required_features).issubset(considered_important_feature_list):
        total_bytes = (
            adversary_bi_flow_stat["flowBytesPerSecond"]
            * adversary_bi_flow_stat["duration"]
            / 10**6
        )
        pkt_list = X_existing_flow_delta_pkt_len
        if num_added_pkts > 0:
            pkt_list = pkt_list + X_added_flow_pkt_len
        s.add(
            z3.And(
                [
                    Sum(pkt_list) + sum(df_benign_bi_flow["pkt_len"])
                    >= lower_bound * total_bytes,
                    Sum(pkt_list) + sum(df_benign_bi_flow["pkt_len"])
                    <= upper_bound * total_bytes,
                ]
            )
        )

    include_active_or_idle = [
        f for f in considered_important_feature_list if "active" in f or "idle" in f
    ]
    if len(include_active_or_idle) != 0 and active_len + idle_len != 1:
        total_num_areas = active_len + idle_len
        num_border_pkts = total_num_areas - 1

        X_border_idx_list = [Int("border_idx%s" % i) for i in range(num_border_pkts)]
        s.add(X_border_idx_list[0] > 0)
        s.add(X_border_idx_list[-1] < num_existing_pkts - 1)
        for i in range(num_border_pkts - 1):
            s.add(X_border_idx_list[i] < X_border_idx_list[i + 1])

        A = Array("A", RealSort(), RealSort())
        i = 0
        for elem in interarrivals:
            A = Store(A, i, elem)
            i = i + 1

        active_idx_list, idle_idx_list = calculate_active_idle_index(
            index, active_threshold, asset_common_dir
        )
        if 0 in active_idx_list[0]:
            active_first = True
        elif 0 in idle_idx_list[0]:
            active_first = False
        else:
            print("ERROR")
            exit(1)

        # If active, record 1, if idle, record 0
        if active_first:
            active_idle_order = [
                1 if i % 2 == 0 else 0
                for i in range(len(active_idx_list) + len(idle_idx_list))
            ]
        else:
            active_idle_order = [
                0 if i % 2 == 0 else 1
                for i in range(len(active_idx_list) + len(idle_idx_list))
            ]

        for idx, is_active in enumerate(active_idle_order):
            if idx == 0:
                if is_active:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                i < X_border_idx_list[idx], A[i] <= active_threshold
                            )
                        )
                else:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                i < X_border_idx_list[idx], A[i] >= active_threshold
                            )
                        )
            elif idx == len(active_idle_order) - 1:
                if is_active:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                i >= X_border_idx_list[idx - 1],
                                A[i] <= active_threshold,
                            )
                        )
                else:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                i >= X_border_idx_list[idx - 1],
                                A[i] >= active_threshold,
                            )
                        )
            else:
                if is_active:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                z3.And(
                                    i >= X_border_idx_list[idx - 1],
                                    i < X_border_idx_list[idx],
                                ),
                                A[i] <= active_threshold,
                            )
                        )
                else:
                    for i in range(len(interarrivals)):
                        s.add(
                            Implies(
                                z3.And(
                                    i >= X_border_idx_list[idx - 1],
                                    i < X_border_idx_list[idx],
                                ),
                                A[i] >= active_threshold,
                            )
                        )

        active_list = []
        idle_list = []
        for idx, is_active in enumerate(active_idle_order):
            # Initialize the sum as 0
            current_sum = 0

            if idx == 0:
                # Calculate the sum of the subarray
                for i in range(len(interarrivals)):
                    current_sum = If(
                        And(i >= 0, i < X_border_idx_list[0]),
                        current_sum + A[i],
                        current_sum,
                    )
                if is_active:
                    active_list.append(current_sum)
                else:
                    idle_list.append(current_sum)
            elif idx == len(active_idle_order) - 1:
                # Calculate the sum of the subarray
                for i in range(len(interarrivals)):
                    current_sum = If(
                        And(i >= X_border_idx_list[idx - 1], i < len(interarrivals)),
                        current_sum + A[i],
                        current_sum,
                    )
                if is_active:
                    active_list.append(current_sum)
                else:
                    idle_list.append(current_sum)
            else:
                # Calculate the sum of the subarray
                for i in range(len(interarrivals)):
                    current_sum = If(
                        And(
                            i >= X_border_idx_list[idx - 1], i < X_border_idx_list[idx]
                        ),
                        current_sum + A[i],
                        current_sum,
                    )
                if is_active:
                    active_list.append(current_sum)
                else:
                    idle_list.append(current_sum)

        # mean_active
        if "mean_active" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(active_list) / len(active_list)
                        >= lower_bound * adversary_bi_flow_stat["mean_active"],
                        Sum(active_list) / len(active_list)
                        <= upper_bound * adversary_bi_flow_stat["mean_active"],
                    ]
                )
            )

        # min_active
        if "min_active" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        active_list[i] >= lower_bound * adversary_bi_flow_stat["min_active"]
                        for i in range(len(active_list))
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                active_list[i]
                                >= lower_bound * adversary_bi_flow_stat["min_active"],
                                active_list[i]
                                <= upper_bound * adversary_bi_flow_stat["min_active"],
                            ]
                        )
                        for i in range(len(active_list))
                    ]
                )
            )

        #  max_active
        if "max_active" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        active_list[i] <= upper_bound * adversary_bi_flow_stat["max_active"]
                        for i in range(len(active_list))
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                active_list[i]
                                >= lower_bound * adversary_bi_flow_stat["max_active"],
                                active_list[i]
                                <= upper_bound * adversary_bi_flow_stat["max_active"],
                            ]
                        )
                        for i in range(len(active_list))
                    ]
                )
            )

        # std_active
        if "std_active" in considered_important_feature_list:
            std_active_list = [
                (i - Sum(active_list) / len(active_list)) ** 2 for i in active_list
            ]
            s.add(
                z3.And(
                    [
                        Sum(std_active_list) / len(std_active_list)
                        >= lower_bound * adversary_bi_flow_stat["std_active"] ** 2,
                        Sum(std_active_list) / len(std_active_list)
                        <= upper_bound * adversary_bi_flow_stat["std_active"] ** 2,
                    ]
                )
            )

        # mean_idle
        if "mean_idle" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(idle_list) / len(idle_list)
                        >= lower_bound * adversary_bi_flow_stat["mean_idle"],
                        Sum(idle_list) / len(idle_list)
                        <= upper_bound * adversary_bi_flow_stat["mean_idle"],
                    ]
                )
            )

        # min_idle
        if "min_idle" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        idle_list[i] >= lower_bound * adversary_bi_flow_stat["min_idle"]
                        for i in range(len(idle_list))
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                idle_list[i]
                                >= lower_bound * adversary_bi_flow_stat["min_idle"],
                                idle_list[i]
                                <= upper_bound * adversary_bi_flow_stat["min_idle"],
                            ]
                        )
                        for i in range(len(idle_list))
                    ]
                )
            )

        # max_idle
        if "max_idle" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        idle_list[i] <= upper_bound * adversary_bi_flow_stat["max_idle"]
                        for i in range(len(idle_list))
                    ]
                )
            )
            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                idle_list[i]
                                >= lower_bound * adversary_bi_flow_stat["max_idle"],
                                idle_list[i]
                                <= upper_bound * adversary_bi_flow_stat["max_idle"],
                            ]
                        )
                        for i in range(len(idle_list))
                    ]
                )
            )

        # std_idle
        if "std_idle" in considered_important_feature_list:
            std_idle_list = [
                (i - Sum(idle_list) / len(idle_list)) ** 2 for i in idle_list
            ]
            s.add(
                z3.And(
                    [
                        Sum(std_idle_list) / len(std_idle_list)
                        >= lower_bound * adversary_bi_flow_stat["std_idle"] ** 2,
                        Sum(std_idle_list) / len(std_idle_list)
                        <= upper_bound * adversary_bi_flow_stat["std_idle"] ** 2,
                    ]
                )
            )

    success = False
    # Check for a solution
    result = s.check()
    if result == sat:
        solution = s.model()
        modified = {
            "srcip": [],
            "dstip": [],
            "srcport": [],
            "dstport": [],
            "proto": [],
            "iat": [],
            "pkt_len": [],
        }

        ts_solution = []
        for i in range(num_existing_pkts):
            delay = solution[Real("dxts%s" % i)].as_fraction()
            ts_solution.append(
                orig_interarrival_list[i]
                + float(delay.numerator) / float(delay.denominator)
            )
        pkt_len_solution = []
        for i in range(num_existing_pkts):
            delta_pkt_len = solution[Real("dxpl%s" % i)].as_fraction()
            pkt_len_solution.append(
                df_benign_bi_flow["pkt_len"][i]
                + float(delta_pkt_len.numerator) / float(delta_pkt_len.denominator)
            )

        for i in range(num_added_pkts):
            delay = solution[Real("axts%s" % i)].as_fraction()
            ts_solution.append(float(delay.numerator) / float(delay.denominator))
            delta_pkt_len = solution[Real("axpl%s" % i)].as_fraction()
            pkt_len_solution.append(
                float(delta_pkt_len.numerator) / float(delta_pkt_len.denominator)
            )

        for i in range(num_existing_pkts + num_added_pkts):
            if fwd_bwd_indicator[i] == 0:
                modified["srcip"].append(fwd_traffic_five_tuple[0])
                modified["dstip"].append(fwd_traffic_five_tuple[1])
                modified["srcport"].append(fwd_traffic_five_tuple[2])
                modified["dstport"].append(fwd_traffic_five_tuple[3])
                modified["proto"].append(fwd_traffic_five_tuple[4])
            else:
                modified["srcip"].append(bwd_traffic_five_tuple[0])
                modified["dstip"].append(bwd_traffic_five_tuple[1])
                modified["srcport"].append(bwd_traffic_five_tuple[2])
                modified["dstport"].append(bwd_traffic_five_tuple[3])
                modified["proto"].append(bwd_traffic_five_tuple[4])
            modified["iat"].append(np.round(ts_solution[i]))
            modified["pkt_len"].append(pkt_len_solution[i])

        success = True
        df_modified = pd.DataFrame(modified)
        df_modified[["srcport", "dstport"]] = df_modified[
            ["srcport", "dstport"]
        ].astype(int)
    else:
        success = False
        df_modified = None
    tmp_queue.append((success, df_modified))
    return success, df_modified
