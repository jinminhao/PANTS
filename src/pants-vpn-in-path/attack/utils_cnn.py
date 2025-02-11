import torch
import numpy as np
from z3 import *
import pandas as pd

import multiprocessing as mp

from itertools import product

TIMEOUT = 2
first_n_pkts = 400

def count_consecutive_zeros_at_end(lst):
    count = 0
    for num in reversed(lst):
        if num == 0:
            count += 1
        else:
            break
    return count

def get_perturbable_mask(orig_feature):
    iats = orig_feature[0, 0, first_n_pkts:].cpu().numpy()
    unchangeable_pkts = count_consecutive_zeros_at_end(iats)
    # unchangeable_pkts = unchangeable_pkts - 20 if unchangeable_pkts > 20 else 0

    perturbable_mask = torch.ones_like(orig_feature)
    if unchangeable_pkts > 0:
        perturbable_mask[:, :, -unchangeable_pkts:] = 0
        perturbable_mask[:, :, first_n_pkts-unchangeable_pkts:first_n_pkts] = 0
    return perturbable_mask

def get_pgd_adv_features(orig_feature, labels, model, loss, total_repeat_time, iters, alpha, normalized_direction_value, normalized_max_pkt_len, normalized_min_pkt_len):

    perturbable_mask = get_perturbable_mask(orig_feature)

    # pkt_len mask
    perturbable_mask[:, :, :first_n_pkts] = 0

    pgd_adv_features_list = []
    perturbed_feature_ = orig_feature.clone().detach()
    mask = torch.ones_like(perturbed_feature_)
    # mask backward pkts not to be perturbable
    for i in range(first_n_pkts):
        if perturbed_feature_[0, 0, i] <= normalized_direction_value:
            mask[0, 0, i] = 0
            mask[0, 0, first_n_pkts + i] = 0

    perturbed_feature = perturbed_feature_ 
    # + torch.zeros_like(
    #     perturbed_feature_
    # ).normal_(mean=0, std=1e-1)
    k = 0
    for iteration in range(total_repeat_time*iters):
        perturbed_feature.requires_grad = True
        outputs = model(perturbed_feature)
        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        sign_grad = perturbed_feature.grad.sign()

        # iat can only increase
        for i in range(first_n_pkts):
            if mask[0, 0, i+first_n_pkts] == 1 and sign_grad[0, 0, i+first_n_pkts] < 0:
                sign_grad[0, 0, i+first_n_pkts] = 0   

        perturbed_feature = perturbed_feature + alpha * sign_grad * mask * perturbable_mask

        # pkt_len should be in a range.
        # for i in range(perturbed_feature.size(1)):
        #     if perturbed_feature[0, i, 0] > normalized_max_pkt_len:
        #         perturbed_feature[0, i, 0] = normalized_max_pkt_len
        #     if perturbed_feature[0, i, 0] < normalized_min_pkt_len and perturbed_feature[0, i, 0] > normalized_direction_value:
        #         perturbed_feature[0, i, 0] = normalized_min_pkt_len

        perturbed_feature = torch.clamp(perturbed_feature, min=0, max=1).detach_()

        if (iteration + 1) % iters == 0:
            added_features = perturbed_feature.clone().detach()
            pgd_adv_features_list.append(added_features)
    return pgd_adv_features_list

def denormalize(orig_feat, max_iats, min_iats, max_pkt_lens, min_pkt_lens):
    orig_feat[:, :, first_n_pkts:] = orig_feat[:, :, first_n_pkts:] * (max_iats - min_iats) + min_iats
    orig_feat[:, :, :first_n_pkts] = orig_feat[:, :, :first_n_pkts] * (max_pkt_lens - min_pkt_lens) + min_pkt_lens
    orig_feat[:, :, first_n_pkts:] = np.expm1(orig_feat[:, :, first_n_pkts:])
    orig_feat[:, :, :first_n_pkts] = np.expm1(np.abs(orig_feat[:, :, :first_n_pkts])) * np.sign(orig_feat[:, :, :first_n_pkts])
    return orig_feat

def normalize(orig_denorm_feat, max_iats, min_iats, max_pkt_lens, min_pkt_lens):
    orig_denorm_feat[:, :, first_n_pkts:] = np.log1p(orig_denorm_feat[:, :, first_n_pkts:])
    orig_denorm_feat[:, :, :first_n_pkts] = np.log1p(np.abs(orig_denorm_feat[:, :, :first_n_pkts])) * np.sign(orig_denorm_feat[:, :, :first_n_pkts])
    orig_denorm_feat[:, :, first_n_pkts:] = (orig_denorm_feat[:, :, first_n_pkts:] - min_iats) / (max_iats - min_iats)
    orig_denorm_feat[:, :, :first_n_pkts] = (orig_denorm_feat[:, :, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
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

def try_smt(df_denorm_adv_inputs, df_denorm_inputs, max_overhead, sorted_idx, remaining_sum_iats, tmp_queue):
    adv_pkt_len = df_denorm_adv_inputs["pkt_len"]
    adv_iats = df_denorm_adv_inputs["iat"]
    orig_iats = df_denorm_inputs["iat"]
    orig_pkt_len = df_denorm_inputs["pkt_len"]

    maximum_iats = (sum(orig_iats) + remaining_sum_iats)*(1+max_overhead)

    s = Solver()
    X_existing_flow_delta_ts = [Real("dxts%s" % i) for i in range(len(orig_iats))]
    for i in range(len(orig_iats)):
        if orig_pkt_len[i] > 0:
            s.add(X_existing_flow_delta_ts[i] >= 0)
            s.add(orig_iats[i] + X_existing_flow_delta_ts[i] <= 1.01 * adv_iats[i])
        else:
            s.add(X_existing_flow_delta_ts[i] == 0)
    for i in range(len(sorted_idx)-1):
        s.add(X_existing_flow_delta_ts[sorted_idx[i]] >= X_existing_flow_delta_ts[sorted_idx[i+1]])

    s.add(sum(X_existing_flow_delta_ts) <= maximum_iats - sum(orig_iats) - remaining_sum_iats)
    s.add(sum(X_existing_flow_delta_ts) >= 0.98 * (maximum_iats - sum(orig_iats) - remaining_sum_iats))

    success = False

    if s.check() == sat:
        solution = s.model()

        ts_solution = []
        for i in range(len(orig_iats)):
            delay = solution[Real("dxts%s" % i)].as_fraction()
            ts_solution.append(
                orig_iats[i]
                + float(delay.numerator) / float(delay.denominator)
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