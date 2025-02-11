import pandas as pd
import numpy as np
import torch
import math
import shap
import os

dataset_fields = {
    "duration": 0,
    "total_fiat": 1,
    "total_biat": 2,
    "min_fiat": 3,
    "min_biat": 4,
    "max_fiat": 5,
    "max_biat": 6,
    "mean_fiat": 7,
    "mean_biat": 8,
    "flowPktsPerSecond": 9,
    "flowBytesPerSecond": 10,
    "min_flowiat": 11,
    "mean_flowiat": 12,
    "max_flowiat": 13,
    "std_flowiat": 14,
    "min_active": 15,
    "mean_active": 16,
    "max_active": 17,
    "std_active": 18,
    "min_idle": 19,
    "mean_idle": 20,
    "max_idle": 21,
    "std_idle": 22,
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


def calculate_active_idle_len(index, active_threshold, asset_common_dir):
    df = pd.read_csv(os.path.join(asset_common_dir, f"pkt_dir/{index}.csv"))

    flowiat = np.diff(df["time"])
    active_indicator = np.array(flowiat <= active_threshold)
    idle_indicator = np.array(flowiat > active_threshold)
    active = []
    active_subsum = 0
    for i in range(len(active_indicator)):
        if active_indicator[i]:
            active_subsum += flowiat[i]
        else:
            if active_subsum != 0:
                active.append(active_subsum)
            active_subsum = 0
    if active_subsum != 0:
        active.append(active_subsum)
        active_subsum = 0

    idle = []
    idle_subsum = 0
    for i in range(len(idle_indicator)):
        if idle_indicator[i]:
            idle_subsum += flowiat[i]
        else:
            if idle_subsum != 0:
                idle.append(idle_subsum)
            idle_subsum = 0
    if idle_subsum != 0:
        idle.append(idle_subsum)
        idle_subsum = 0

    return len(active), len(idle)


# The function will return the index of interval for active and idle (x, y) should include both x and y
def calculate_active_idle_index(index, active_threshold, asset_common_dir):
    df = pd.read_csv(os.path.join(asset_common_dir, f"pkt_dir/{index}.csv"))
    flowiat = np.diff(df["time"])
    active_indicator = np.array(flowiat <= active_threshold)
    idle_indicator = np.array(flowiat > active_threshold)
    active_idx_list = []
    idle_idx_list = []

    start = None
    for i, element in enumerate(active_indicator):
        if element and start is None:  # Start of a new interval
            start = i
        elif not element and start is not None:  # End of the current interval
            active_idx_list.append((start, i - 1))
            start = None

    # If the last element is True, close the final interval
    if start is not None:
        active_idx_list.append((start, len(active_indicator) - 1))

    start = None
    for i, element in enumerate(idle_indicator):
        if element and start is None:  # Start of a new interval
            start = i
        elif not element and start is not None:  # End of the current interval
            idle_idx_list.append((start, i - 1))
            start = None

    # If the last element is True, close the final interval
    if start is not None:
        idle_idx_list.append((start, len(idle_indicator) - 1))

    return active_idx_list, idle_idx_list


def duration_loss(outputs, active_len, idle_len, scaler):
    min_val = torch.from_numpy(scaler.data_min_)
    max_val = torch.from_numpy(scaler.data_max_)
    torch_denorm_outputs = min_val + (max_val - min_val) * outputs
    torch_denorm_outputs = torch.expm1(torch_denorm_outputs)

    duration_1 = (
        active_len * torch_denorm_outputs[:, dataset_fields["mean_active"]]
        + idle_len * torch_denorm_outputs[:, dataset_fields["mean_idle"]]
    )
    num_fwd_pkts = (
        torch_denorm_outputs[:, dataset_fields["total_fiat"]]
        / torch_denorm_outputs[:, dataset_fields["mean_fiat"]]
        + 1
    )
    num_bwd_pkts = (
        torch_denorm_outputs[:, dataset_fields["total_biat"]]
        / torch_denorm_outputs[:, dataset_fields["mean_biat"]]
        + 1
    )

    duration_2 = (
        (num_fwd_pkts + num_bwd_pkts)
        / torch_denorm_outputs[:, dataset_fields["flowPktsPerSecond"]]
        * 10**6
    )

    duration = torch_denorm_outputs[:, dataset_fields["duration"]]
    return 0.5 * (
        torch.norm(duration_2 - duration_1) + torch.norm(duration_1 - duration)
    )


def transfer_to_features(bi_flow, active_threshold):
    five_tuples = ["srcip", "dstip", "srcport", "dstport", "proto"]
    bi_flow = bi_flow.sort_values(by=["time"]).reset_index(drop=True)

    processed_df_dict = {
        # "srcip": [],
        # "dstip": [],
        # "srcport": [],
        # "dstport": [],
        # "proto": [],
        "duration": [],
        "total_fiat": [],
        "total_biat": [],
        "min_fiat": [],
        "min_biat": [],
        "max_fiat": [],
        "max_biat": [],
        "mean_fiat": [],
        "mean_biat": [],
        "flowPktsPerSecond": [],
        "flowBytesPerSecond": [],
        "min_flowiat": [],
        "mean_flowiat": [],
        "max_flowiat": [],
        "std_flowiat": [],
        "min_active": [],
        "mean_active": [],
        "max_active": [],
        "std_active": [],
        "min_idle": [],
        "mean_idle": [],
        "max_idle": [],
        "std_idle": [],
    }

    grouped = bi_flow.groupby(five_tuples)

    biat = None
    srcip = None
    dstip = None
    srcport = None
    dstport = None
    proto = None
    duration = np.max(bi_flow["time"]) - np.min(bi_flow["time"])
    # Remove flows where duration is 0
    if duration == 0:
        print("ERROR: Duration is 0")
        return pd.DataFrame(processed_df_dict)
    fb_psec = np.sum(bi_flow["pkt_len"]) / (duration / (10**6))
    fp_psec = len(bi_flow) / (duration / (10**6))

    flowiat = np.diff(bi_flow["time"])
    flowiat_min = np.min(flowiat)
    flowiat_mean = np.mean(flowiat)
    flowiat_max = np.max(flowiat)
    flowiat_std = np.std(flowiat)

    active_indicator = np.array(flowiat <= active_threshold)
    idle_indicator = np.array(flowiat > active_threshold)

    active = []
    active_subsum = 0
    for i in range(len(active_indicator)):
        if active_indicator[i]:
            active_subsum += flowiat[i]
        else:
            if active_subsum != 0:
                active.append(active_subsum)
            active_subsum = 0
    if active_subsum != 0:
        active.append(active_subsum)
        active_subsum = 0

    idle = []
    idle_subsum = 0
    for i in range(len(idle_indicator)):
        if idle_indicator[i]:
            idle_subsum += flowiat[i]
        else:
            if idle_subsum != 0:
                idle.append(idle_subsum)
            idle_subsum = 0
    if idle_subsum != 0:
        idle.append(idle_subsum)
        idle_subsum = 0

    if len(active) > 0:
        active_mean = np.mean(active)
        active_std = np.std(active)
        active_max = np.max(active)
        active_min = np.min(active)
    else:
        active_mean = 0
        active_std = 0
        active_max = 0
        active_min = 0

    if len(idle) > 0:
        idle_mean = np.mean(idle)
        idle_std = np.std(idle)
        idle_max = np.max(idle)
        idle_min = np.min(idle)
    else:
        idle_mean = 0
        idle_std = 0
        idle_max = 0
        idle_min = 0

    invalid = False
    for name, group in grouped:
        if len(group) == 1:
            invalid = True
    if invalid:
        print("ERROR: Invalid flow")
        return pd.DataFrame(processed_df_dict)

    fwd_traffic_five_tuple = list(bi_flow.iloc[0][five_tuples])
    bwd_traffic_five_tuple = [
        fwd_traffic_five_tuple[1],
        fwd_traffic_five_tuple[0],
        fwd_traffic_five_tuple[3],
        fwd_traffic_five_tuple[2],
        fwd_traffic_five_tuple[4],
    ]
    for name, group in grouped:
        if list(name) == fwd_traffic_five_tuple:
            fiat = np.diff(group["time"])
            fiat_total = np.sum(fiat)
            fiat_mean = np.mean(fiat)
            fiat_std = np.std(fiat)
            fiat_max = np.max(fiat)
            fiat_min = np.min(fiat)
            srcip, dstip, srcport, dstport, proto = (
                name[0],
                name[1],
                name[2],
                name[3],
                name[4],
            )
        elif list(name) == bwd_traffic_five_tuple:
            biat = np.diff(group["time"])
            biat_total = np.sum(biat)
            biat_mean = np.mean(biat)
            biat_std = np.std(biat)
            biat_max = np.max(biat)
            biat_min = np.min(biat)
        else:
            print("ERROR: Invalid five tuple")
            exit(1)

    if len(grouped) == 1:
        biat_total = -1
        biat_mean = -1
        biat_std = -1
        biat_max = -1
        biat_min = -1
        print("ERROR: Only one direction flow")
        return pd.DataFrame(processed_df_dict)

    # processed_df_dict["srcip"].append(srcip)
    # processed_df_dict["dstip"].append(dstip)
    # processed_df_dict["srcport"].append(srcport)
    # processed_df_dict["dstport"].append(dstport)
    # processed_df_dict["proto"].append(proto)

    processed_df_dict["duration"].append(duration)

    processed_df_dict["total_fiat"].append(fiat_total)
    processed_df_dict["total_biat"].append(biat_total)

    processed_df_dict["min_fiat"].append(fiat_min)
    processed_df_dict["min_biat"].append(biat_min)

    processed_df_dict["max_fiat"].append(fiat_max)
    processed_df_dict["max_biat"].append(biat_max)

    processed_df_dict["mean_fiat"].append(fiat_mean)
    processed_df_dict["mean_biat"].append(biat_mean)

    # processed_df_dict["std_fiat"].append(fiat_std)
    # processed_df_dict["std_biat"].append(biat_std)

    processed_df_dict["flowPktsPerSecond"].append(fp_psec)
    processed_df_dict["flowBytesPerSecond"].append(fb_psec)

    processed_df_dict["min_flowiat"].append(flowiat_min)
    processed_df_dict["mean_flowiat"].append(flowiat_mean)
    processed_df_dict["max_flowiat"].append(flowiat_max)
    processed_df_dict["std_flowiat"].append(flowiat_std)

    processed_df_dict["min_active"].append(active_min)
    processed_df_dict["mean_active"].append(active_mean)
    processed_df_dict["max_active"].append(active_max)
    processed_df_dict["std_active"].append(active_std)

    processed_df_dict["min_idle"].append(idle_min)
    processed_df_dict["mean_idle"].append(idle_mean)
    processed_df_dict["max_idle"].append(idle_max)
    processed_df_dict["std_idle"].append(idle_std)

    # print(processed_df_dict)

    return pd.DataFrame(processed_df_dict)


def mask_feat(sign_grad, only_src):
    mask = torch.ones_like(sign_grad)
    if only_src:
        if sign_grad[:, dataset_fields["min_biat"]][0] < 0:
            mask[:, dataset_fields["min_biat"]] = 0
        if sign_grad[:, dataset_fields["max_biat"]][0] < 0:
            mask[:, dataset_fields["max_biat"]] = 0
    else:
        if sign_grad[:, dataset_fields["min_fiat"]][0] < 0:
            mask[:, dataset_fields["min_fiat"]] = 0
        if sign_grad[:, dataset_fields["max_fiat"]][0] < 0:
            mask[:, dataset_fields["max_fiat"]] = 0

    if sign_grad[:, dataset_fields["mean_biat"]][0] < 0:
        mask[:, dataset_fields["mean_biat"]] = 0
    if sign_grad[:, dataset_fields["total_fiat"]][0] < 0:
        mask[:, dataset_fields["total_fiat"]] = 0

    # mask[:, dataset_fields["max_active"]] = 0
    # mask[:, dataset_fields["min_active"]] = 0
    # mask[:, dataset_fields["mean_active"]] = 0
    # mask[:, dataset_fields["std_active"]] = 0
    # mask[:, dataset_fields["max_idle"]] = 0
    # mask[:, dataset_fields["min_idle"]] = 0
    # mask[:, dataset_fields["mean_idle"]] = 0
    # mask[:, dataset_fields["std_idle"]] = 0
    return mask


def get_important_features(X_train, mlp_classifier, inputs):
    background = torch.tensor(X_train).type(torch.FloatTensor)
    e = shap.DeepExplainer(mlp_classifier, background)
    shap_values = e.shap_values(inputs)
    shap_values = np.array(shap_values)[0][:]
    # abs_shap_values = np.abs(shap_values)
    # indice_sort = np.argsort(-abs_shap_values)

    sum_shap_values = np.sum(shap_values)

    if sum_shap_values < 0:
        shap_values = -shap_values
    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    # tmp = [x for x in tmp if "active" not in x]
    # tmp = [x for x in tmp if "idle" not in x]
    return tmp, indice_sort


def get_important_features_svm(X_train, svm_classifier, inputs, K=10):
    background = X_train
    e = shap.KernelExplainer(
        svm_classifier.predict_proba, shap.sample(background, K), link="logit"
    )
    shap_values = e.shap_values(inputs, silent=True)
    label = svm_classifier.predict(inputs)[0]
    shap_values = np.array(shap_values)[int(label)][0][:]
    abs_shap_values = np.abs(shap_values)
    # indice_sort = np.argsort(-abs_shap_values)

    # sum_shap_values = np.sum(shap_values)
    # if sum_shap_values < 0:
    #     shap_values = -shap_values
    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    tmp = [x for x in tmp if "active" not in x]
    tmp = [x for x in tmp if "idle" not in x]
    return tmp, indice_sort


def get_important_features_v2(inputs, refs):
    diff = np.abs(inputs[0] - refs[0])

    indice_sort = np.argsort(-diff)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    return tmp, indice_sort


def get_important_features_xgb(X_train, xgb_classifier, inputs, K=100):
    background = X_train
    e = shap.TreeExplainer(xgb_classifier, shap.sample(background, K), link="logit")
    shap_values = e.shap_values(inputs.reshape(1, -1))
    shap_values = np.array(shap_values)[0][:]

    abs_shap_values = np.abs(shap_values)
    # indice_sort = np.argsort(-abs_shap_values)

    sum_shap_values = np.sum(shap_values)
    if sum_shap_values < 0:
        shap_values = -shap_values
    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    # tmp = [x for x in tmp if "active" not in x]
    # tmp = [x for x in tmp if "idle" not in x]
    return tmp, indice_sort


def get_important_features_rf(X_train, xgb_classifier, inputs, label, K=100):
    background = X_train
    e = shap.TreeExplainer(xgb_classifier, shap.sample(background, K), link="logit")
    shap_values = e.shap_values(inputs.reshape(1, -1))
    shap_values = np.array(shap_values)[int(label)][0][:]

    abs_shap_values = np.abs(shap_values)
    # indice_sort = np.argsort(-abs_shap_values)

    # sum_shap_values = np.sum(shap_values)
    # if sum_shap_values < 0:
    #     shap_values = -shap_values
    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    # tmp = [x for x in tmp if "active" not in x]
    # tmp = [x for x in tmp if "idle" not in x]
    return tmp, indice_sort


def self_correct_biat(perturbed_features, ori_features, scaler):
    denorm_perturbed_features = data_denormalization(perturbed_features, scaler)
    denorm_ori_features = data_denormalization(ori_features, scaler)

    if (
        denorm_perturbed_features[0][dataset_fields["min_biat"]]
        < denorm_ori_features[0][dataset_fields["min_biat"]]
    ):
        denorm_perturbed_features[0][dataset_fields["min_biat"]] = denorm_ori_features[
            0
        ][dataset_fields["min_biat"]]
    if (
        denorm_perturbed_features[0][dataset_fields["max_biat"]]
        < denorm_ori_features[0][dataset_fields["max_biat"]]
    ):
        denorm_perturbed_features[0][dataset_fields["max_biat"]] = denorm_ori_features[
            0
        ][dataset_fields["max_biat"]]
    if (
        denorm_perturbed_features[0][dataset_fields["mean_biat"]]
        < denorm_ori_features[0][dataset_fields["mean_biat"]]
    ):
        denorm_perturbed_features[0][dataset_fields["mean_biat"]] = denorm_ori_features[
            0
        ][dataset_fields["mean_biat"]]
    if (
        denorm_perturbed_features[0][dataset_fields["total_biat"]]
        < denorm_ori_features[0][dataset_fields["total_biat"]]
    ):
        denorm_perturbed_features[0][dataset_fields["total_biat"]] = (
            denorm_ori_features[0][dataset_fields["total_biat"]]
        )

    return data_normalization(denorm_perturbed_features, scaler)


# Solve the constraint issue on the important feature
def self_correct_pkt_num(
    perturbed_features,
    scaler,
    important_feature_list,
    important_feature_indices_list,
    orig_fwd_pkts,
    orig_bwd_pkts,
):
    torch_denorm_perturbed_features = torch_data_denormalization(
        perturbed_features, scaler
    )

    # number of fwd pkts should be larger or equal to the original value
    num_fwd_pkts = (
        torch.round(
            torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]]
            / torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]]
        )
        + 1
    )

    if num_fwd_pkts <= orig_fwd_pkts:
        # modify less important ones
        idx_total_fiat = important_feature_list.index("total_fiat")
        idx_mean_fiat = important_feature_list.index("mean_fiat")
        if idx_total_fiat < idx_mean_fiat:
            # total_fiat is more important
            torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]] = (
                torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]]
                / (orig_fwd_pkts - 1)
            )
        elif idx_total_fiat > idx_mean_fiat:
            # mean_fiat is more important
            torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]] = (
                orig_fwd_pkts - 1
            ) * (torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]])

    # if num_fwd_pkts - orig_fwd_pkts > 10:
    #     # modify less important ones
    #     idx_total_fiat = important_feature_list.index("total_fiat")
    #     idx_mean_fiat = important_feature_list.index("mean_fiat")
    #     if idx_total_fiat < idx_mean_fiat:
    #         # total_fiat is more important
    #         torch_denorm_perturbed_features[
    #             :, dataset_fields["mean_fiat"]
    #         ] = torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]] / (
    #             orig_fwd_pkts + 10 - 1
    #         )
    #     elif idx_total_fiat > idx_mean_fiat:
    #         # mean_fiat is more important
    #         torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]] = (
    #             orig_fwd_pkts + 10 - 1
    #         ) * (torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]])

    # number of bwd pkts should be equal to the original value
    num_bwd_pkts = (
        torch.round(
            torch_denorm_perturbed_features[:, dataset_fields["total_biat"]]
            / torch_denorm_perturbed_features[:, dataset_fields["mean_biat"]]
        )
        + 1
    )

    if num_bwd_pkts != orig_bwd_pkts:
        # modify less important ones
        idx_total_biat = important_feature_list.index("total_biat")
        idx_mean_biat = important_feature_list.index("mean_biat")
        if idx_total_biat < idx_mean_biat:
            # total_biat is more important
            torch_denorm_perturbed_features[:, dataset_fields["mean_biat"]] = (
                torch_denorm_perturbed_features[:, dataset_fields["total_biat"]]
                / (orig_bwd_pkts - 1)
            )
        elif idx_total_biat > idx_mean_biat:
            # mean_biat is more important
            torch_denorm_perturbed_features[:, dataset_fields["total_biat"]] = (
                orig_bwd_pkts - 1
            ) * (torch_denorm_perturbed_features[:, dataset_fields["mean_biat"]])

    torch_renorm_perturbed_features = torch_data_normalization(
        torch_denorm_perturbed_features, scaler
    )
    return torch_renorm_perturbed_features
