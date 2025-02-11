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
# OVERHEAD = 0.28
# PKT_PAYLOAD_APPEND_LIMIT = 0.25
# MAX_NUM_PKTS = 0


def call_smt(
    mode,  # ["single", "chunks"]
    df_stat,
    considered_important_feature_list,
    considered_important_feature_indices_list,
    perturb_src,
    asset_common_dir,
    OVERHEAD,
    PKT_PAYLOAD_APPEND_LIMIT,
    MAX_NUM_PKTS,
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
        "total_biat" in considered_important_feature_list
        and "mean_biat" in considered_important_feature_list
    ):
        num_adversary_bwd_pkts = int(
            np.round(
                adversary_bi_flow_stat["total_biat"]
                / adversary_bi_flow_stat["mean_biat"]
            )
            + 1
        )
        num_benign_bwd_pkts = int(
            np.round(
                benign_bi_flow_stat["total_biat"] / benign_bi_flow_stat["mean_biat"]
            )
            + 1
        )
        num_added_pkts = num_adversary_bwd_pkts - num_benign_bwd_pkts
        if num_added_pkts < 0:
            num_added_pkts = 0
        if int(num_added_pkts) > MAX_NUM_PKTS:
            num_added_pkts = MAX_NUM_PKTS
        num_added_pkts_list = [int(num_added_pkts)]

    elif "total_bpkt" in considered_important_feature_list:
        num_added_pkts = np.round(adversary_bi_flow_stat["total_bpkt"]) - np.round(
            benign_bi_flow_stat["total_bpkt"]
        )
        if num_added_pkts < 0:
            num_added_pkts = 0
        if int(num_added_pkts) > MAX_NUM_PKTS:
            num_added_pkts = MAX_NUM_PKTS
        num_added_pkts_list = [int(num_added_pkts)]

    elif (
        "total_fiat" in considered_important_feature_list
        and "mean_fiat" in considered_important_feature_list
    ):
        num_adversary_fwd_pkts = int(
            np.round(
                adversary_bi_flow_stat["total_fiat"]
                / adversary_bi_flow_stat["mean_fiat"]
            )
            + 1
        )
        num_benign_fwd_pkts = int(
            np.round(
                benign_bi_flow_stat["total_fiat"] / benign_bi_flow_stat["mean_fiat"]
            )
            + 1
        )
        num_added_pkts = num_adversary_fwd_pkts - num_benign_fwd_pkts
        if num_added_pkts < 0:
            num_added_pkts = 0
        if int(num_added_pkts) > MAX_NUM_PKTS:
            num_added_pkts = MAX_NUM_PKTS
        num_added_pkts_list = [int(num_added_pkts)]

    elif "total_fpkt" in considered_important_feature_list:
        num_added_pkts = np.round(adversary_bi_flow_stat["total_fpkt"]) - np.round(
            benign_bi_flow_stat["total_fpkt"]
        )
        if num_added_pkts < 0:
            num_added_pkts = 0
        if int(num_added_pkts) > MAX_NUM_PKTS:
            num_added_pkts = MAX_NUM_PKTS
        num_added_pkts_list = [int(num_added_pkts)]

    else:
        num_added_pkts_list = [i for i in range(0, MAX_NUM_PKTS+1)]

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
                OVERHEAD,
                PKT_PAYLOAD_APPEND_LIMIT,
                MAX_NUM_PKTS,
                tolerance,
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
        df_stat_updated.loc[0, "total_fbyt"] /= num_iter
        df_stat_updated.loc[0, "total_bbyt"] /= num_iter
        for idx in range(num_iter):
            df_this_chunk = orig_df[
                idx * chunk_size : (idx + 1) * chunk_size
            ].reset_index(drop=True)

            tmp_chunk_df_results = [df_this_chunk]
            # We need to make sure the chunk can be at least in length 1
            # This should be commented out for strong capability
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
                        OVERHEAD,
                        PKT_PAYLOAD_APPEND_LIMIT,
                        MAX_NUM_PKTS,
                        tolerance,
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


def try_smt_strong(
    df_stat,
    df_flow,
    considered_important_feature_list,
    num_added_pkts,
    num_splitted_pkts_percent,
    perturb_src,
    fwd_traffic_five_tuple,
    OVERHEAD,
    PKT_PAYLOAD_APPEND_LIMIT,
    MAX_NUM_PKTS,
    tolerance,
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
        pkt_appendable_idx = [
            i for i in range(num_existing_pkts) if fwd_bwd_indicator[i] == 0
        ]
    else:
        pkt_appendable_idx = [
            i for i in range(num_existing_pkts) if fwd_bwd_indicator[i] == 1
        ]
    random.shuffle(pkt_appendable_idx)
    tmp_len = len(pkt_appendable_idx)
    pkt_appendable_idx = pkt_appendable_idx[: int(PKT_PAYLOAD_APPEND_LIMIT * tmp_len)]

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
    flow_pkt_len = []
    for i in range(num_existing_pkts):
        if i == 0:
            flow_abs_ts.append(orig_interarrival_list[i] + X_existing_flow_delta_ts[i])
        else:
            flow_abs_ts.append(
                flow_abs_ts[-1]
                + orig_interarrival_list[i]
                + X_existing_flow_delta_ts[i]
            )
        flow_pkt_len.append(
            df_benign_bi_flow["pkt_len"][i] + X_existing_flow_delta_pkt_len[i]
        )
    for i in range(num_added_pkts):
        flow_abs_ts.append(flow_abs_ts[-1] + X_added_flow_delta_ts[i])
        flow_pkt_len.append(X_added_flow_pkt_len[i])
    s.add(
        (flow_abs_ts[-1] - flow_abs_ts[0])
        <= (1 + OVERHEAD) * sum(orig_interarrival_list)
    )

    fwd_indices = [i for i, x in enumerate(fwd_bwd_indicator) if x == 0]
    if len(fwd_indices) >= 2:
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

        # std_fiat
        if "std_fiat" in considered_important_feature_list:
            tmp_list = [
                flow_abs_ts[fwd_indices[i + 1]] - flow_abs_ts[fwd_indices[i]]
                for i in range(len(fwd_indices) - 1)
            ]
            std_fiat = [(i - (Sum(tmp_list)) / len(tmp_list)) ** 2 for i in tmp_list]
            s.add(
                z3.And(
                    [
                        sum(std_fiat) / len(tmp_list)
                        <= upper_bound * adversary_bi_flow_stat["std_fiat"] ** 2,
                        sum(std_fiat) / len(tmp_list)
                        >= lower_bound * adversary_bi_flow_stat["std_fiat"] ** 2,
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
                        * len(bwd_indices),
                        flow_abs_ts[bwd_indices[-1]] - flow_abs_ts[bwd_indices[0]]
                        <= upper_bound
                        * adversary_bi_flow_stat["mean_biat"]
                        * len(bwd_indices),
                    ]
                )
            )

        # std_biat
        if "std_biat" in considered_important_feature_list:
            tmp_list = [
                flow_abs_ts[bwd_indices[i + 1]] - flow_abs_ts[bwd_indices[i]]
                for i in range(len(bwd_indices) - 1)
            ]
            std_biat = [(i - (Sum(tmp_list)) / len(tmp_list)) ** 2 for i in tmp_list]
            s.add(
                z3.And(
                    [
                        sum(std_biat) / len(tmp_list)
                        <= upper_bound * adversary_bi_flow_stat["std_biat"] ** 2,
                        sum(std_biat) / len(tmp_list)
                        >= lower_bound * adversary_bi_flow_stat["std_biat"] ** 2,
                    ]
                )
            )

    fbyt_list = [
        flow_pkt_len[i] for i in range(len(flow_pkt_len)) if fwd_bwd_indicator[i] == 0
    ]
    existing_fbyt_list = [
        df_benign_bi_flow["pkt_len"][i]
        for i in range(len(df_benign_bi_flow["pkt_len"]))
        if fwd_bwd_indicator[i] == 0 and i < num_existing_pkts
    ]

    if len(fbyt_list) > 0:
        # total_fbyt
        if "total_fbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(fbyt_list)
                        <= upper_bound * adversary_bi_flow_stat["total_fbyt"],
                        Sum(fbyt_list)
                        >= lower_bound * adversary_bi_flow_stat["total_fbyt"],
                    ]
                )
            )

        # min_fbyt
        if "min_fbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        fbyt_list[i] >= lower_bound * adversary_bi_flow_stat["min_fbyt"]
                        for i in range(len(fbyt_list))
                    ]
                )
            )

            if (
                len(existing_fbyt_list) > 0 and min(existing_fbyt_list)
                > upper_bound * adversary_bi_flow_stat["min_fbyt"]
            ):
                s.add(
                    z3.Or(
                        [
                            z3.And(
                                [
                                    fbyt_list[i]
                                    >= lower_bound * adversary_bi_flow_stat["min_fbyt"],
                                    fbyt_list[i]
                                    <= upper_bound * adversary_bi_flow_stat["min_fbyt"],
                                ]
                            )
                            for i in range(len(fbyt_list))
                        ]
                    )
                )

        # max_fbyt
        if "max_fbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        fbyt_list[i] <= upper_bound * adversary_bi_flow_stat["max_fbyt"]
                        for i in range(len(fbyt_list))
                    ]
                )
            )

            if (
                len(existing_fbyt_list) > 0 and max(existing_fbyt_list)
                < lower_bound * adversary_bi_flow_stat["max_fbyt"]
            ):
                s.add(
                    z3.Or(
                        [
                            z3.And(
                                [
                                    fbyt_list[i]
                                    >= lower_bound * adversary_bi_flow_stat["max_fbyt"],
                                    fbyt_list[i]
                                    <= upper_bound * adversary_bi_flow_stat["max_fbyt"],
                                ]
                            )
                            for i in range(len(fbyt_list))
                        ]
                    )
                )

        # mean_fbyt
        if "mean_fbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(fbyt_list)
                        <= upper_bound
                        * adversary_bi_flow_stat["mean_fbyt"]
                        * len(fbyt_list),
                        Sum(fbyt_list)
                        >= lower_bound
                        * adversary_bi_flow_stat["mean_fbyt"]
                        * len(fbyt_list),
                    ]
                )
            )

        # std_fbyt
        if "std_fbyt" in considered_important_feature_list:
            tmp_list = fbyt_list
            std_fbyt = [(i - (Sum(fbyt_list)) / len(tmp_list)) ** 2 for i in tmp_list]
            s.add(
                z3.And(
                    [
                        sum(std_fbyt) / len(tmp_list)
                        <= upper_bound * adversary_bi_flow_stat["std_fbyt"] ** 2,
                        sum(std_fbyt) / len(tmp_list)
                        >= lower_bound * adversary_bi_flow_stat["std_fbyt"] ** 2,
                    ]
                )
            )

    bbyt_list = [
        flow_pkt_len[i] for i in range(len(flow_pkt_len)) if fwd_bwd_indicator[i] == 1
    ]
    existing_bbyt_list = [
        df_benign_bi_flow["pkt_len"][i]
        for i in range(len(df_benign_bi_flow["pkt_len"]))
        if fwd_bwd_indicator[i] == 1 and i < num_existing_pkts
    ]

    if len(bbyt_list) > 0:
        # total_bbyt
        if "total_bbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(bbyt_list)
                        <= upper_bound * adversary_bi_flow_stat["total_bbyt"],
                        Sum(bbyt_list)
                        >= lower_bound * adversary_bi_flow_stat["total_bbyt"],
                    ]
                )
            )

        # min_bbyt
        if "min_bbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        bbyt_list[i] >= lower_bound * adversary_bi_flow_stat["min_bbyt"]
                        for i in range(len(bbyt_list))
                    ]
                )
            )

            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                bbyt_list[i]
                                >= lower_bound * adversary_bi_flow_stat["min_bbyt"],
                                bbyt_list[i]
                                <= upper_bound * adversary_bi_flow_stat["min_bbyt"],
                            ]
                        )
                        for i in range(len(bbyt_list))
                    ]
                )
            )

        # max_bbyt
        if "max_bbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        bbyt_list[i] <= upper_bound * adversary_bi_flow_stat["max_bbyt"]
                        for i in range(len(bbyt_list))
                    ]
                )
            )

            s.add(
                z3.Or(
                    [
                        z3.And(
                            [
                                bbyt_list[i]
                                >= lower_bound * adversary_bi_flow_stat["max_bbyt"],
                                bbyt_list[i]
                                <= upper_bound * adversary_bi_flow_stat["max_bbyt"],
                            ]
                        )
                        for i in range(len(bbyt_list))
                    ]
                )
            )

        # mean_bbyt
        if "mean_bbyt" in considered_important_feature_list:
            s.add(
                z3.And(
                    [
                        Sum(bbyt_list)
                        <= upper_bound
                        * adversary_bi_flow_stat["mean_bbyt"]
                        * len(bbyt_list),
                        Sum(bbyt_list)
                        >= lower_bound
                        * adversary_bi_flow_stat["mean_bbyt"]
                        * len(bbyt_list),
                    ]
                )
            )

        # std_bbyt
        if "std_bbyt" in considered_important_feature_list:
            tmp_list = bbyt_list
            std_bbyt = [(i - (Sum(bbyt_list)) / len(tmp_list)) ** 2 for i in tmp_list]
            s.add(
                z3.And(
                    [
                        sum(std_bbyt) / len(tmp_list)
                        <= upper_bound * adversary_bi_flow_stat["std_bbyt"] ** 2,
                        sum(std_bbyt) / len(tmp_list)
                        >= lower_bound * adversary_bi_flow_stat["std_bbyt"] ** 2,
                    ]
                )
            )

    success = False
    # Check for a solution
    if s.check() == sat:
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
            modified["pkt_len"].append(int(pkt_len_solution[i]))

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
