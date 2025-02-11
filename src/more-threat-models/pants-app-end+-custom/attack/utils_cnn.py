import torch
import numpy as np
from z3 import *
import pandas as pd

import multiprocessing as mp

from itertools import product

TIMEOUT = 2
# PKT_PAYLOAD_APPEND_LIMIT = 0.2
first_n_pkts = 400
# MAX_ADDED_PKTS = 20


def count_consecutive_zeros_at_end(lst):
    count = 0
    for num in reversed(lst):
        if num == 0:
            count += 1
        else:
            break
    return count


def get_perturbable_mask(orig_feature, orig_feature_len):
    iats = orig_feature[0, 0, first_n_pkts:].cpu().numpy()
    unchangeable_pkts = 400 - orig_feature_len
    # unchangeable_pkts = unchangeable_pkts - 20 if unchangeable_pkts > 20 else 0

    perturbable_mask = torch.ones_like(orig_feature)
    if unchangeable_pkts > 0:
        perturbable_mask[:, :, -unchangeable_pkts:] = 0
        perturbable_mask[:, :, first_n_pkts - unchangeable_pkts : first_n_pkts] = 0
    return perturbable_mask


def check_existing_fwd_pkt(added_feature, normalized_min_pkt_len):
    out = []
    for i in range(first_n_pkts):
        if added_feature[0, 0, i] >= normalized_min_pkt_len:
            out.append(i)
    return out


def check_all_fwd_pkt(added_feature, normalized_direction_value):
    out = []
    for i in range(first_n_pkts):
        if added_feature[0, 0, i] >= normalized_direction_value:
            out.append(i)
    return out


def find_top_largest_indices(lst, top_n):
    # Pair each value with its index
    indexed_lst = list(enumerate(lst))

    # Sort the list by the values in descending order
    sorted_indexed_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)

    # Extract the indices of the top_n largest values
    top_indices = [index for index, value in sorted_indexed_lst[:top_n]]

    return top_indices


def pgd_func(
    prepared_feature,
    prepared_feature_len,
    injected_pkt_idx_list,
    existing_fwd_pkt_idx,
    splitted_flow_idx,
    splitted_pkt_size,
    normalized_direction_value,
    normalized_min_pkt_len,
    total_repeat_time,
    iters,
    model,
    loss,
    labels,
    alpha,
    PKT_PAYLOAD_APPEND_LIMIT,
    tmp_queue,
):
    prepared_feature = torch.tensor(prepared_feature).float()
    perturbable_mask = get_perturbable_mask(prepared_feature, prepared_feature_len)
    fwd_pkt_idx = []
    # pkt_len mask
    perturbable_pkt_size_idx = [
        i
        for i in range(first_n_pkts)
        if prepared_feature[0, 0, i] > normalized_direction_value
    ]
    perturbable_mask[:, :, :first_n_pkts] = 0
    perturbable_mask[:, :, perturbable_pkt_size_idx] = 1

    perturbed_feature_ = prepared_feature.clone().detach()
    mask = torch.ones_like(perturbed_feature_)
    # mask backward pkts not to be perturbable
    for i in range(first_n_pkts):
        if perturbed_feature_[0, 0, i] <= normalized_direction_value:
            mask[0, 0, i] = 0
            mask[0, 0, first_n_pkts + i] = 0
        else:
            fwd_pkt_idx.append(i)

    perturbed_feature = perturbed_feature_
    # + torch.zeros_like(
    #     perturbed_feature_
    # ).normal_(mean=0, std=1e-1)
    k = 0
    for iteration in range(total_repeat_time * iters):
        perturbed_feature.requires_grad = True
        outputs = model(perturbed_feature)
        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        sign_grad = perturbed_feature.grad.sign()

        # iat can only increase
        for i in range(first_n_pkts):
            if i in fwd_pkt_idx and sign_grad[0, 0, i + first_n_pkts] < 0:
                sign_grad[0, 0, i + first_n_pkts] = 0
            # pkt_len can only increase
            if i in fwd_pkt_idx and sign_grad[0, 0, i] < 0:
                sign_grad[0, 0, i] = 0

        perturbed_feature = (
            perturbed_feature + alpha * sign_grad * mask * perturbable_mask
        )
        perturbed_feature[:, :, splitted_flow_idx] = torch.clamp(perturbed_feature[:, :, splitted_flow_idx], min=normalized_min_pkt_len)

        # pkt_len should be in a range.
        # for i in range(perturbed_feature.size(1)):
        #     if perturbed_feature[0, i, 0] > normalized_max_pkt_len:
        #         perturbed_feature[0, i, 0] = normalized_max_pkt_len
        #     if perturbed_feature[0, i, 0] < normalized_min_pkt_len and perturbed_feature[0, i, 0] > normalized_direction_value:
        #         perturbed_feature[0, i, 0] = normalized_min_pkt_len

        perturbed_feature = torch.clamp(perturbed_feature, min=0, max=1).detach_()

        if (iteration + 1) % iters == 0:
            added_features = perturbed_feature.clone().detach()
            diff = (
                added_features[0, 0, existing_fwd_pkt_idx]
                - prepared_feature[0, 0, existing_fwd_pkt_idx]
            )
            idx_list = find_top_largest_indices(
                diff, int(PKT_PAYLOAD_APPEND_LIMIT * len(existing_fwd_pkt_idx))
            )
            allowed_to_change_payload_idx = [existing_fwd_pkt_idx[i] for i in idx_list]
            not_allowd_to_change_payload_idx = [
                i
                for i in existing_fwd_pkt_idx
                if i not in allowed_to_change_payload_idx
            ]
            added_features[0, 0, not_allowd_to_change_payload_idx] = prepared_feature[
                0, 0, not_allowd_to_change_payload_idx
            ]
            tmp_queue.append(
                (added_features, prepared_feature_len, injected_pkt_idx_list)
            )


def get_pgd_adv_features(
    orig_feature,
    labels,
    orig_flow_len,
    model,
    loss,
    total_repeat_time,
    iters,
    alpha,
    normalized_direction_value,
    normalized_max_pkt_len,
    normalized_min_pkt_len,
    PKT_PAYLOAD_APPEND_LIMIT,
    MAX_ADDED_PKTS,
    PKT_SPLIT_LIMIT
):
    flow_length = orig_flow_len
    sample_num = 10
    large_candidate_list = []
    
    num_orig_forward_pkts = len([i for i in range(first_n_pkts) if orig_feature[0, 0, i] > normalized_direction_value])
    max_num_split_pkts = int(np.round(num_orig_forward_pkts * PKT_SPLIT_LIMIT))
    num_orig_forward_pkts_candidate = [i for i in range(1, max_num_split_pkts+1)]
    if len(num_orig_forward_pkts_candidate) >= 2:
        num_splitted_pkts_list = list(np.random.choice(num_orig_forward_pkts_candidate, 2, replace=False)) + [0]
    else:
        num_splitted_pkts_list = [0]

    idx_fwd_pkt = [i for i in range(first_n_pkts) if orig_feature[0, 0, i] > normalized_direction_value]
    
    splitted_flows_list = []
    splitted_flow_idx_list = []
    splitted_flow_length_list = []
    splitted_pkt_size_list = []
    for num_splitted_pkts in num_splitted_pkts_list:
        split_pkt_idx_list = np.random.choice(idx_fwd_pkt, num_splitted_pkts, replace=False)
        split_pkt_idx_list.sort()
        split_pkt_idx_list = [idx + i for i, idx in enumerate(split_pkt_idx_list)]
        split_pkt_idx_list = [i for i in split_pkt_idx_list if i < first_n_pkts]
        splitted_flow_length = flow_length + len(split_pkt_idx_list)
        splitted_flow_idx_list.append(split_pkt_idx_list)
        splitted_pkt_size_list.append([orig_feature[0, 0, i].item() for i in split_pkt_idx_list])
        splitted_flow_length_list.append(splitted_flow_length)
        
        prev_position = 0
        chunk_pkt_len = []
        chunk_iat = []
        for position in split_pkt_idx_list:
            chunk_pkt_len.append(orig_feature[:, :, prev_position:position])
            dummy_pkt_len = np.zeros((1, 1, 1))
            dummy_pkt_len[0, 0, 0] = normalized_min_pkt_len-0.0001
            chunk_pkt_len.append(dummy_pkt_len) 
            chunk_iat.append(orig_feature[:, :, (first_n_pkts + prev_position):(first_n_pkts + position)])
            dummy_iat = np.zeros((1, 1, 1))
            dummy_iat[0, 0, 0] = 0.0001
            chunk_iat.append(dummy_iat)
            prev_position = position
            
        chunk_pkt_len.append(orig_feature[:, :, prev_position:first_n_pkts])
        chunk_iat.append(orig_feature[:, :, (first_n_pkts + prev_position):(first_n_pkts + first_n_pkts)])
        
        pkt_len = np.concatenate(chunk_pkt_len, axis=2)[:, :, :first_n_pkts]
        appended_pkt_len = np.ones((1, 1, first_n_pkts-pkt_len.shape[2])) * normalized_direction_value
        pkt_len = np.concatenate([pkt_len, appended_pkt_len], axis=2)
        iat = np.concatenate(chunk_iat, axis=2)[:, :, :first_n_pkts]
        appended_iat = np.zeros((1, 1, first_n_pkts-iat.shape[2]))
        iat = np.concatenate([iat, appended_iat], axis=2)
        new_flow = np.concatenate((pkt_len, iat), axis=2)
        splitted_flows_list.append(new_flow)
    
    for splitted_flow, splitted_flow_len,  splitted_flow_idx, splitted_pkt_size in zip(splitted_flows_list, splitted_flow_length_list, splitted_flow_idx_list, splitted_pkt_size_list):
        added_pkt_num_candidate_1 = [i for i in range(15, MAX_ADDED_PKTS+1)]
        added_pkt_num_candidate_2 = [i for i in range(0, MAX_ADDED_PKTS+1)]
        added_pkt_num_list = list(np.random.choice(added_pkt_num_candidate_1, int(sample_num/2), replace=True)) + list(np.random.choice(added_pkt_num_candidate_2, int(sample_num/2), replace=True))
        for added_pkt_num in added_pkt_num_list:
            position_list = np.random.choice(range(1, flow_length+1), added_pkt_num, replace=True)
            position_list.sort()
            
            splitted_flow_idx = [i + len([x for x in position_list if x < i]) for i in splitted_flow_idx]

            injected_pkt_idx_list = [i + position_list[i] for i in range(added_pkt_num)]
            injected_pkt_idx_list = [i for i in injected_pkt_idx_list if i < first_n_pkts]
            prev_position = 0
            chunk_pkt_len = []
            chunk_iat = []
            for position in position_list:
                chunk_pkt_len.append(orig_feature[:, :, prev_position:position])
                dummy_pkt_len = np.zeros((1, 1, 1))
                dummy_pkt_len[0, 0, 0] = normalized_min_pkt_len-0.0001
                chunk_pkt_len.append(dummy_pkt_len) 
                chunk_iat.append(orig_feature[:, :, (first_n_pkts + prev_position):(first_n_pkts + position)])
                dummy_iat = np.zeros((1, 1, 1))
                dummy_iat[0, 0, 0] = 0.0001
                chunk_iat.append(dummy_iat)
                prev_position = position

            chunk_pkt_len.append(orig_feature[:, :, prev_position:first_n_pkts])
            chunk_iat.append(orig_feature[:, :, (first_n_pkts + prev_position):(first_n_pkts + first_n_pkts)])
            
            pkt_len = np.concatenate(chunk_pkt_len, axis=2)[:, :, :first_n_pkts]
            appended_pkt_len = np.ones((1, 1, first_n_pkts-pkt_len.shape[2])) * normalized_direction_value
            pkt_len = np.concatenate([pkt_len, appended_pkt_len], axis=2)
            iat = np.concatenate(chunk_iat, axis=2)[:, :, :first_n_pkts]
            appended_iat = np.zeros((1, 1, first_n_pkts-iat.shape[2]))
            iat = np.concatenate([iat, appended_iat], axis=2)
            new_flow = np.concatenate((pkt_len, iat), axis=2)
            large_candidate_list.append((new_flow, orig_flow_len+added_pkt_num, injected_pkt_idx_list))

    pgd_adv_features_list = []
    for feature, feature_len, injected_pkt_idx_list in large_candidate_list:

        existing_fwd_pkt_idx = check_existing_fwd_pkt(feature, normalized_min_pkt_len)
        # all_fwd_pkt_idx = check_all_fwd_pkt(feature, normalized_direction_value)
        
        pgd_func(feature, feature_len, injected_pkt_idx_list, existing_fwd_pkt_idx, splitted_flow_idx, splitted_pkt_size, normalized_direction_value, normalized_min_pkt_len, total_repeat_time, iters, model, loss, labels, alpha, PKT_PAYLOAD_APPEND_LIMIT, pgd_adv_features_list)
    return pgd_adv_features_list


def denormalize(orig_feat, max_iats, min_iats, max_pkt_lens, min_pkt_lens):
    orig_feat[:, :, first_n_pkts:] = (
        orig_feat[:, :, first_n_pkts:] * (max_iats - min_iats) + min_iats
    )
    orig_feat[:, :, :first_n_pkts] = (
        orig_feat[:, :, :first_n_pkts] * (max_pkt_lens - min_pkt_lens) + min_pkt_lens
    )
    orig_feat[:, :, first_n_pkts:] = np.expm1(orig_feat[:, :, first_n_pkts:])
    orig_feat[:, :, :first_n_pkts] = np.expm1(
        np.abs(orig_feat[:, :, :first_n_pkts])
    ) * np.sign(orig_feat[:, :, :first_n_pkts])
    return orig_feat


def normalize(orig_denorm_feat, max_iats, min_iats, max_pkt_lens, min_pkt_lens):
    orig_denorm_feat[:, :, first_n_pkts:] = np.log1p(
        orig_denorm_feat[:, :, first_n_pkts:]
    )
    orig_denorm_feat[:, :, :first_n_pkts] = np.log1p(
        np.abs(orig_denorm_feat[:, :, :first_n_pkts])
    ) * np.sign(orig_denorm_feat[:, :, :first_n_pkts])
    orig_denorm_feat[:, :, first_n_pkts:] = (
        orig_denorm_feat[:, :, first_n_pkts:] - min_iats
    ) / (max_iats - min_iats)
    orig_denorm_feat[:, :, :first_n_pkts] = (
        orig_denorm_feat[:, :, :first_n_pkts] - min_pkt_lens
    ) / (max_pkt_lens - min_pkt_lens)
    return orig_denorm_feat


def get_important_features(df_denorm_adv_inputs, df_denorm_inputs):
    adv_iats = df_denorm_adv_inputs["iat"]
    orig_iats = df_denorm_inputs["iat"]

    orig_pkt_len = df_denorm_inputs["pkt_len"]
    perturbable_idx = [i for i in range(len(orig_pkt_len)) if orig_pkt_len[i] > 0]

    diff = list(np.abs(adv_iats - orig_iats))

    def indices_sorted_by_values_desc(lst):
        return sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)

    def filter_list_by_another(list_to_filter, reference_list):
        return [item for item in list_to_filter if item in reference_list]

    sorted_idx = indices_sorted_by_values_desc(diff)
    sorted_idx = filter_list_by_another(sorted_idx, perturbable_idx)

    return sorted_idx


def call_smt(smt_process_list):
    tmp = smt_group_per_iter(smt_process_list, try_smt)
    out = []
    for success, df_modified in tmp:
        if success:
            out.append(df_modified)
    return out


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


def try_smt(
    df_denorm_adv_inputs,
    df_denorm_inputs,
    max_overhead,
    sorted_idx,
    remaining_sum_iats,
    tmp_queue,
):
    adv_pkt_len = df_denorm_adv_inputs["pkt_len"]
    adv_iats = df_denorm_adv_inputs["iat"]
    orig_iats = df_denorm_inputs["iat"]
    orig_pkt_len = df_denorm_inputs["pkt_len"]

    maximum_iats = (sum(orig_iats) + remaining_sum_iats) * (1 + max_overhead)

    s = Solver()
    X_existing_flow_delta_ts = [Real("dxts%s" % i) for i in range(len(orig_iats))]
    for i in range(len(orig_iats)):
        if orig_pkt_len[i] > 0:
            s.add(X_existing_flow_delta_ts[i] >= 0)
            s.add(
                orig_iats[i] + X_existing_flow_delta_ts[i] <= max(1.01 * adv_iats[i], 1)
            )
        else:
            s.add(X_existing_flow_delta_ts[i] == 0)
    for i in range(len(sorted_idx) - 1):
        s.add(
            X_existing_flow_delta_ts[sorted_idx[i]]
            >= X_existing_flow_delta_ts[sorted_idx[i + 1]]
        )

    s.add(
        sum(X_existing_flow_delta_ts)
        <= maximum_iats - sum(orig_iats) - remaining_sum_iats
    )
    s.add(
        sum(X_existing_flow_delta_ts)
        >= 0.98 * (maximum_iats - sum(orig_iats) - remaining_sum_iats)
    )

    success = False

    if s.check() == sat:
        solution = s.model()

        ts_solution = []
        for i in range(len(orig_iats)):
            delay = solution[Real("dxts%s" % i)].as_fraction()
            ts_solution.append(
                orig_iats[i] + float(delay.numerator) / float(delay.denominator)
            )

        modified = {}
        modified["iat"] = ts_solution
        modified["pkt_len"] = adv_pkt_len
        df_modified = pd.DataFrame(modified, columns=["pkt_len", "iat"])
        success = True
    else:
        success = False
        df_modified = None

    tmp_queue.append((success, df_modified))
    return success, df_modified
