import pandas as pd
import numpy as np
import torch
import math
import shap

dataset_fields = {
    "l_max": 0,
    "l_min": 1,
    "l_mean": 2,
    "l_std": 3,
    "l_q2": 4,
    "t_min": 5,
    "t_max": 6,
    "t_mean": 7,
    "t_std": 8,
    "t_q2": 9,
    "l_num_bytes": 10,
    "l_num_pkts": 11,
    "l_num_unique": 12,
    "t_burst_count": 13,
}


def data_denormalization(adv_sample, scaler):
    denorm_adv_sample = scaler.inverse_transform(adv_sample)
    denorm_adv_sample = np.expm1(denorm_adv_sample)
    return denorm_adv_sample


def data_normalization(adv_sample, scaler):
    if isinstance(adv_sample, pd.DataFrame):
        adv_sample = adv_sample.to_numpy()
    norm_adv_sample = np.log1p(adv_sample)
    norm_adv_sample = scaler.transform(norm_adv_sample)
    return norm_adv_sample


def torch_data_denormalization(adv_sample, scaler):
    min_val = torch.from_numpy(scaler.data_min_)
    max_val = torch.from_numpy(scaler.data_max_)
    torch_denorm_adv_sample = min_val + (max_val - min_val) * adv_sample
    torch_denorm_adv_sample = torch.expm1(torch_denorm_adv_sample)
    return torch_denorm_adv_sample


def torch_data_normalization(adv_sample, scaler):
    min_val = torch.from_numpy(scaler.data_min_)
    max_val = torch.from_numpy(scaler.data_max_)
    torch_norm_adv_sample = torch.log1p(adv_sample)
    torch_norm_adv_sample = (torch_norm_adv_sample - min_val) / (max_val - min_val)
    return torch_norm_adv_sample.float()


############################################################################
#                      Constraint function definitions                     #
############################################################################


def min_mean_max_constraint(min, max, mean):
    if isinstance(min, pd.Series):
        min = min[0]
    if isinstance(max, pd.Series):
        max = max[0]
    if isinstance(mean, pd.Series):
        mean = mean[0]
    return True if (min <= mean and mean <= max) else False


def std_constraint(min, max, std):
    if isinstance(min, pd.Series):
        min = min[0]
    if isinstance(max, pd.Series):
        max = max[0]
    if isinstance(std, pd.Series):
        std = std[0]
    return True if ((std <= (max - min)) and (0 <= std)) else False


def lt_eq_constraint(x, y):
    if isinstance(x, pd.Series):
        x = x[0]
    if isinstance(y, pd.Series):
        y = y[0]
    return x <= y


def duration_loss(outputs, scaler):
    min_val = torch.from_numpy(scaler.data_min_)
    max_val = torch.from_numpy(scaler.data_max_)
    torch_denorm_outputs = min_val + (max_val - min_val) * outputs
    torch_denorm_outputs = torch.expm1(torch_denorm_outputs)

    num_pkts_1 = torch_denorm_outputs[:, dataset_fields["l_num_pkts"]]
    num_pkts_2 = (
        torch_denorm_outputs[:, dataset_fields["l_num_bytes"]]
        / torch_denorm_outputs[:, dataset_fields["l_mean"]]
    )

    return torch.norm(num_pkts_1 - num_pkts_2)


def min_max_loss(outputs, scaler):
    min_val = torch.from_numpy(scaler.data_min_)
    max_val = torch.from_numpy(scaler.data_max_)
    torch_denorm_outputs = min_val + (max_val - min_val) * outputs
    torch_denorm_outputs = torch.expm1(torch_denorm_outputs)

    l_min = torch_denorm_outputs[:, dataset_fields["l_min"]]
    l_max = torch_denorm_outputs[:, dataset_fields["l_max"]]
    l_mean = torch_denorm_outputs[:, dataset_fields["l_mean"]]
    l_q2 = torch_denorm_outputs[:, dataset_fields["l_q2"]]
    term0 = torch.norm((l_max + l_min) / 2 - l_q2)
    term1 = torch.norm((l_max + l_min) / 2 - l_mean)
    term2 = l_max - l_min

    l_loss = -term0 - term1 + term2

    t_min = torch_denorm_outputs[:, dataset_fields["t_min"]]
    t_max = torch_denorm_outputs[:, dataset_fields["t_max"]]
    t_mean = torch_denorm_outputs[:, dataset_fields["t_mean"]]
    t_q2 = torch_denorm_outputs[:, dataset_fields["t_q2"]]
    term0 = torch.norm((t_max + t_min) / 2 - t_q2)
    term1 = torch.norm((t_max + t_min) / 2 - t_mean)
    term2 = t_max - t_min

    t_loss = -term0 - term1 + term2

    return l_loss + t_loss


def transfer_to_features(flow):
    sum_flow_time = 0
    last_pkt_idx = len(flow)
    for i in range(len(flow)):
        if i == 0:
            continue
        sum_flow_time += flow["iats"][i]
        if sum_flow_time > 1000:
            last_pkt_idx = i
            break

    flow = flow[:last_pkt_idx]

    processed_df_dict = {
        # "srcip": [],
        # "dstip": [],
        # "srcport": [],
        # "dstport": [],
        # "proto": [],
        "l_max": [],
        "l_min": [],
        "l_mean": [],
        "l_std": [],
        "l_q2": [],
        "t_min": [],
        "t_max": [],
        "t_mean": [],
        "t_std": [],
        "t_q2": [],
        "l_num_bytes": [],
        "l_num_pkts": [],
        "l_num_unique": [],
        "t_burst_count": [],
    }

    # HERE: add the features

    l_max = np.max(flow["length"])
    l_min = np.min(flow["length"])
    l_mean = np.mean(flow["length"])
    l_std = np.std(flow["length"])
    l_q2 = np.quantile(flow["length"], 0.5)

    flowiat = flow["iats"]
    t_max = np.max(flowiat)
    t_min = np.min(flowiat)
    t_mean = np.mean(flowiat)
    t_std = np.std(flowiat)
    t_q2 = np.quantile(flowiat, 0.5)

    l_num_bytes = np.sum(flow["length"])
    l_num_pkts = len(flow)
    l_num_unique = len(flow["length"].unique())

    def calculate_burst_count(x):
        x = np.array(x)
        if len(x) <= 1:
            return 0
        mask = x >= 30
        return mask.sum()

    t_burst_count = calculate_burst_count(flowiat)

    processed_df_dict["l_max"].append(l_max)
    processed_df_dict["l_min"].append(l_min)
    processed_df_dict["l_mean"].append(l_mean)
    processed_df_dict["l_std"].append(l_std)
    processed_df_dict["l_q2"].append(l_q2)
    processed_df_dict["t_min"].append(t_min)
    processed_df_dict["t_max"].append(t_max)
    processed_df_dict["t_mean"].append(t_mean)
    processed_df_dict["t_std"].append(t_std)
    processed_df_dict["t_q2"].append(t_q2)
    processed_df_dict["l_num_bytes"].append(l_num_bytes)
    processed_df_dict["l_num_pkts"].append(l_num_pkts)
    processed_df_dict["l_num_unique"].append(l_num_unique)
    processed_df_dict["t_burst_count"].append(t_burst_count)

    return pd.DataFrame(processed_df_dict)


def mask_feat(sign_grad):
    mask = torch.ones_like(sign_grad)
    if sign_grad[:, dataset_fields["l_num_bytes"]][0] < 0:
        mask[:, dataset_fields["l_num_bytes"]] = 0

    if sign_grad[:, dataset_fields["l_num_pkts"]][0] < 0:
        mask[:, dataset_fields["l_num_pkts"]] = 0

    if sign_grad[:, dataset_fields["l_max"]][0] < 0:
        mask[:, dataset_fields["l_max"]] = 0
    # mask[:, dataset_fields["l_num_pkts"]] = 0
    return mask


def get_important_features(X_train, mlp_classifier, inputs, K=1000):
    background = torch.tensor(X_train).type(torch.FloatTensor)
    e = shap.DeepExplainer(mlp_classifier, shap.sample(background, K))
    shap_values = e.shap_values(inputs)
    shap_values = np.array(shap_values)[0][0][:]

    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]

    # tmp = [x for x in tmp if "unique" not in x]
    # tmp = [x for x in tmp if "std" not in x]

    return tmp, indice_sort


def get_important_features_v2(inputs, refs):
    diff = np.abs(inputs[0] - refs[0])

    indice_sort = np.argsort(-diff)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    return tmp, indice_sort


def get_important_features_svm(X_train, svm_classifier, inputs, K=100):
    background = X_train
    e = shap.KernelExplainer(svm_classifier.predict_proba, shap.sample(background, K))
    shap_values = e.shap_values(inputs)
    shap_values = np.array(shap_values)[0][:]
    # abs_shap_values = np.abs(shap_values)
    # indice_sort = np.argsort(-abs_shap_values)

    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    # tmp = [x for x in tmp if "active" not in x]
    # tmp = [x for x in tmp if "idle" not in x]
    return tmp, indice_sort


# Solve the constraint issue on the important feature
def self_correct_pkt_num(
    perturbed_features,
    scaler,
    important_feature_list,
    important_feature_indices_list,
    orig_feature,
):
    torch_denorm_perturbed_features = torch_data_denormalization(
        perturbed_features, scaler
    )

    orig_pkts = np.round(orig_feature["l_num_pkts"][0])
    orig_t_min = orig_feature["t_min"][0]
    orig_t_max = orig_feature["t_max"][0]
    orig_t_mean = orig_feature["t_mean"][0]
    orig_t_a2 = orig_feature["t_q2"][0]

    # number of fwd pkts should be larger or equal to the original value
    num_pkts = torch_denorm_perturbed_features[:, dataset_fields["l_num_pkts"]]

    if num_pkts <= orig_pkts:
        torch_denorm_perturbed_features[:, dataset_fields["l_num_pkts"]] = orig_pkts

    # if num_pkts > orig_pkts + 5:
    #     torch_denorm_perturbed_features[:, dataset_fields["l_num_pkts"]] = orig_pkts + 5

    torch_denorm_perturbed_features[:, dataset_fields["t_min"]] = torch.clip(
        torch_denorm_perturbed_features[:, dataset_fields["t_min"]],
        min=0.9 * orig_t_min,
        max=1.1 * orig_t_min,
    )
    torch_denorm_perturbed_features[:, dataset_fields["t_max"]] = torch.clip(
        torch_denorm_perturbed_features[:, dataset_fields["t_max"]],
        min=0.9 * orig_t_max,
        max=1.1 * orig_t_max,
    )
    torch_denorm_perturbed_features[:, dataset_fields["t_mean"]] = torch.clip(
        torch_denorm_perturbed_features[:, dataset_fields["t_mean"]],
        min=0.9 * orig_t_mean,
        max=1.1 * orig_t_mean,
    )
    torch_denorm_perturbed_features[:, dataset_fields["t_q2"]] = torch.clip(
        torch_denorm_perturbed_features[:, dataset_fields["t_q2"]],
        min=0.9 * orig_t_a2,
        max=1.1 * orig_t_a2,
    )

    torch_renorm_perturbed_features = torch_data_normalization(
        torch_denorm_perturbed_features, scaler
    )
    return torch_renorm_perturbed_features
