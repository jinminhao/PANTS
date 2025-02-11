import pandas as pd
import numpy as np
import torch
import math
import shap

dataset_fields = {
    "total_fiat": 0,
    "total_biat": 1,
    "min_fiat": 2,
    "min_biat": 3,
    "max_fiat": 4,
    "max_biat": 5,
    "mean_fiat": 6,
    "mean_biat": 7,
    "std_fiat": 8,
    "std_biat": 9,
    "total_fpkt": 10,
    "total_bpkt": 11,
    "total_fbyt": 12,
    "total_bbyt": 13,
    "min_fbyt": 14,
    "min_bbyt": 15,
    "max_fbyt": 16,
    "max_bbyt": 17,
    "mean_fbyt": 18,
    "mean_bbyt": 19,
    "std_fbyt": 20,
    "std_bbyt": 21,
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

    fwd_num_pkts = torch_denorm_outputs[:, dataset_fields["total_fpkt"]]
    fwd_num_pkts_1 = (
        torch_denorm_outputs[:, dataset_fields["total_fiat"]]
        / torch_denorm_outputs[:, dataset_fields["mean_fiat"]]
        + 1
    )

    bwd_num_pkts = torch_denorm_outputs[:, dataset_fields["total_bpkt"]]
    bwd_num_pkts_1 = (
        torch_denorm_outputs[:, dataset_fields["total_biat"]]
        / torch_denorm_outputs[:, dataset_fields["mean_biat"]]
        + 1
    )

    return 0.5 * (
        torch.norm(fwd_num_pkts - fwd_num_pkts_1)
        + torch.norm(bwd_num_pkts - bwd_num_pkts_1)
    )


def transfer_to_features(bi_flow):
    five_tuples = ["srcip", "dstip", "srcport", "dstport", "proto"]
    bi_flow = bi_flow.sort_values(by=["time"]).reset_index(drop=True)

    processed_df_dict = {
        # "srcip": [],
        # "dstip": [],
        # "srcport": [],
        # "dstport": [],
        # "proto": [],
        "total_fiat": [],
        "total_biat": [],
        "min_fiat": [],
        "min_biat": [],
        "max_fiat": [],
        "max_biat": [],
        "mean_fiat": [],
        "mean_biat": [],
        "std_fiat": [],
        "std_biat": [],
        "total_fpkt": [],
        "total_bpkt": [],
        "total_fbyt": [],
        "total_bbyt": [],
        "min_fbyt": [],
        "min_bbyt": [],
        "max_fbyt": [],
        "max_bbyt": [],
        "mean_fbyt": [],
        "mean_bbyt": [],
        "std_fbyt": [],
        "std_bbyt": [],
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
            fpkt_total = len(group)
            fbyt_total = np.sum(group["pkt_len"])
            fbyt_mean = np.mean(group["pkt_len"])
            fbyt_std = np.std(group["pkt_len"])
            fbyt_max = np.max(group["pkt_len"])
            fbyt_min = np.min(group["pkt_len"])
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
            bpkt_total = len(group)
            bbyt_total = np.sum(group["pkt_len"])
            bbyt_mean = np.mean(group["pkt_len"])
            bbyt_std = np.std(group["pkt_len"])
            bbyt_max = np.max(group["pkt_len"])
            bbyt_min = np.min(group["pkt_len"])
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

    processed_df_dict["total_fiat"].append(fiat_total)
    processed_df_dict["total_biat"].append(biat_total)

    processed_df_dict["min_fiat"].append(fiat_min)
    processed_df_dict["min_biat"].append(biat_min)

    processed_df_dict["max_fiat"].append(fiat_max)
    processed_df_dict["max_biat"].append(biat_max)

    processed_df_dict["mean_fiat"].append(fiat_mean)
    processed_df_dict["mean_biat"].append(biat_mean)

    processed_df_dict["std_fiat"].append(fiat_std)
    processed_df_dict["std_biat"].append(biat_std)

    processed_df_dict["total_fpkt"].append(fpkt_total)
    processed_df_dict["total_bpkt"].append(bpkt_total)

    processed_df_dict["total_fbyt"].append(fbyt_total)
    processed_df_dict["total_bbyt"].append(bbyt_total)

    processed_df_dict["min_fbyt"].append(fbyt_min)
    processed_df_dict["min_bbyt"].append(bbyt_min)

    processed_df_dict["max_fbyt"].append(fbyt_max)
    processed_df_dict["max_bbyt"].append(bbyt_max)

    processed_df_dict["mean_fbyt"].append(fbyt_mean)
    processed_df_dict["mean_bbyt"].append(bbyt_mean)

    processed_df_dict["std_fbyt"].append(fbyt_std)
    processed_df_dict["std_bbyt"].append(bbyt_std)

    # print(processed_df_dict)

    return pd.DataFrame(processed_df_dict)


def mask_feat(sign_grad):
    mask = torch.ones_like(sign_grad)


    if sign_grad[0, dataset_fields["total_biat"]] < 0:
        mask[:, dataset_fields["total_biat"]] = 0

    if sign_grad[0, dataset_fields["min_biat"]] > 0:
        mask[:, dataset_fields["min_biat"]] = 0

    if sign_grad[0, dataset_fields["max_biat"]] < 0:
        mask[:, dataset_fields["max_biat"]] = 0

    if sign_grad[0, dataset_fields["total_bpkt"]] < 0:
        mask[:, dataset_fields["total_bpkt"]] = 0

    if sign_grad[0, dataset_fields["total_bbyt"]] < 0:
        mask[:, dataset_fields["total_bbyt"]] = 0

    if sign_grad[0, dataset_fields["min_bbyt"]] > 0:
        mask[:, dataset_fields["min_bbyt"]] = 0

    if sign_grad[0, dataset_fields["max_bbyt"]] < 0:
        mask[:, dataset_fields["max_bbyt"]] = 0
    return mask


def get_important_features(X_train, mlp_classifier, inputs):
    background = torch.tensor(X_train).type(torch.FloatTensor)
    e = shap.DeepExplainer(mlp_classifier, background)
    shap_values = e.shap_values(inputs)
    label = torch.argmax(mlp_classifier(inputs)[0]).detach().numpy()

    # shap_values = np.array(shap_values)[0][0][:]
    shap_values = np.array(shap_values)[int(label)][0][:]

    indice_sort = np.argsort(-shap_values)

    fields = list(dataset_fields.keys())
    tmp = [fields[i] for i in indice_sort]
    # Should be removed??
    tmp = [x for x in tmp if "total_fpkt" not in x]
    tmp = [x for x in tmp if "total_fbyt" not in x]
    tmp = [x for x in tmp if "min_fbyt" not in x]
    tmp = [x for x in tmp if "max_fbyt" not in x]
    tmp = [x for x in tmp if "mean_fbyt" not in x]
    tmp = [x for x in tmp if "std_fbyt" not in x]

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

    # if num_fwd_pkts <= orig_fwd_pkts:
    #     # modify less important ones
    #     idx_total_fiat = important_feature_list.index("total_fiat")
    #     idx_mean_fiat = important_feature_list.index("mean_fiat")
    #     if idx_total_fiat < idx_mean_fiat:
    #         # total_fiat is more important
    #         torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]] = (
    #             torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]]
    #             / (orig_fwd_pkts - 1)
    #         )
    #     elif idx_total_fiat > idx_mean_fiat:
    #         # mean_fiat is more important
    #         torch_denorm_perturbed_features[:, dataset_fields["total_fiat"]] = (
    #             orig_fwd_pkts - 1
    #         ) * (torch_denorm_perturbed_features[:, dataset_fields["mean_fiat"]])

    # number of bwd pkts should be equal to the original value
    num_bwd_pkts = (
        torch.round(
            torch_denorm_perturbed_features[:, dataset_fields["total_biat"]]
            / torch_denorm_perturbed_features[:, dataset_fields["mean_biat"]]
        )
        + 1
    )

    if num_bwd_pkts <= orig_bwd_pkts:
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
            # mean_fiat is more important
            torch_denorm_perturbed_features[:, dataset_fields["total_biat"]] = (
                orig_bwd_pkts - 1
            ) * (torch_denorm_perturbed_features[:, dataset_fields["mean_biat"]])

    torch_renorm_perturbed_features = torch_data_normalization(
        torch_denorm_perturbed_features, scaler
    )
    return torch_renorm_perturbed_features

def count_consecutive_zeros_at_end(lst):
    count = 0
    for num in reversed(lst):
        if num == 0:
            count += 1
        else:
            break
    return count

def cast_to_tf_format(flow):
    five_tuple = ["srcip", "dstip", "srcport", "dstport", "proto"]
    first_n_pkts = 400
    flow.sort_values(by="time", inplace=True)
    flow_ndarray = []
    fwd_five_tuple = list(flow.iloc[0][five_tuple])
    bwd_five_tuple = [
        fwd_five_tuple[1],
        fwd_five_tuple[0],
        fwd_five_tuple[3],
        fwd_five_tuple[2],
        fwd_five_tuple[4],
    ]

    iat = np.diff(flow["time"])
    iat = np.insert(iat, 0, 0)

    for index, row in flow.iterrows():
        if list(row[five_tuple]) == fwd_five_tuple:
            flow_ndarray.append([row["pkt_len"], iat[index]])
        elif list(row[five_tuple]) == bwd_five_tuple:
            flow_ndarray.append([-row["pkt_len"], iat[index]])
        else:
            print("Error")
        # if row["pkt_len"] > 1500:
        #     print(flow)
        #     exit()
    
    remaining_pkts = first_n_pkts - len(flow_ndarray) if len(flow_ndarray) < first_n_pkts else 0

    if remaining_pkts > 0:
        for _ in range(remaining_pkts):
            flow_ndarray.append([0, 0])
    return np.array(flow_ndarray)

def tf_format_to_feature(bi_flow):

    bi_flow["pkt_len"] = abs(bi_flow["pkt_len"])
    bi_flow = bi_flow.sort_values(by=["time"]).reset_index(drop=True)

    processed_df_dict = {
        # "srcip": [],
        # "dstip": [],
        # "srcport": [],
        # "dstport": [],
        # "proto": [],
        "total_fiat": [],
        "total_biat": [],
        "min_fiat": [],
        "min_biat": [],
        "max_fiat": [],
        "max_biat": [],
        "mean_fiat": [],
        "mean_biat": [],
        "std_fiat": [],
        "std_biat": [],
        "total_fpkt": [],
        "total_bpkt": [],
        "total_fbyt": [],
        "total_bbyt": [],
        "min_fbyt": [],
        "min_bbyt": [],
        "max_fbyt": [],
        "max_bbyt": [],
        "mean_fbyt": [],
        "mean_bbyt": [],
        "std_fbyt": [],
        "std_bbyt": [],
    }

    grouped = bi_flow.groupby(["fwd"])

    biat = None
    srcip = None
    dstip = None
    srcport = None
    dstport = None
    proto = None
    duration = np.max(bi_flow["time"]) - np.min(bi_flow["time"])
    # Remove flows where duration is 0
    if duration == 0:
        # print("ERROR: Duration is 0")
        return pd.DataFrame(processed_df_dict)

    invalid = False
    for name, group in grouped:
        if len(group) == 1:
            invalid = True
    if invalid:
        # print("ERROR: Invalid flow")
        return pd.DataFrame(processed_df_dict)

    for name, group in grouped:
        if name == True:
            fiat = np.diff(group["time"])
            fiat_total = np.sum(fiat)
            fiat_mean = np.mean(fiat)
            fiat_std = np.std(fiat)
            fiat_max = np.max(fiat)
            fiat_min = np.min(fiat)
            fpkt_total = len(group)
            fbyt_total = np.sum(group["pkt_len"])
            fbyt_mean = np.mean(group["pkt_len"])
            fbyt_std = np.std(group["pkt_len"])
            fbyt_max = np.max(group["pkt_len"])
            fbyt_min = np.min(group["pkt_len"])
        elif name == False:
            biat = np.diff(group["time"])
            biat_total = np.sum(biat)
            biat_mean = np.mean(biat)
            biat_std = np.std(biat)
            biat_max = np.max(biat)
            biat_min = np.min(biat)
            bpkt_total = len(group)
            bbyt_total = np.sum(group["pkt_len"])
            bbyt_mean = np.mean(group["pkt_len"])
            bbyt_std = np.std(group["pkt_len"])
            bbyt_max = np.max(group["pkt_len"])
            bbyt_min = np.min(group["pkt_len"])
        else:
            print(name)
            print("ERROR: Invalid five tuple")
            exit(1)

    if len(grouped) == 1:
        biat_total = -1
        biat_mean = -1
        biat_std = -1
        biat_max = -1
        biat_min = -1
        # print("ERROR: Only one direction flow")
        return pd.DataFrame(processed_df_dict)

    # processed_df_dict["srcip"].append(srcip)
    # processed_df_dict["dstip"].append(dstip)
    # processed_df_dict["srcport"].append(srcport)
    # processed_df_dict["dstport"].append(dstport)
    # processed_df_dict["proto"].append(proto)

    processed_df_dict["total_fiat"].append(fiat_total)
    processed_df_dict["total_biat"].append(biat_total)

    processed_df_dict["min_fiat"].append(fiat_min)
    processed_df_dict["min_biat"].append(biat_min)

    processed_df_dict["max_fiat"].append(fiat_max)
    processed_df_dict["max_biat"].append(biat_max)

    processed_df_dict["mean_fiat"].append(fiat_mean)
    processed_df_dict["mean_biat"].append(biat_mean)

    processed_df_dict["std_fiat"].append(fiat_std)
    processed_df_dict["std_biat"].append(biat_std)

    processed_df_dict["total_fpkt"].append(fpkt_total)
    processed_df_dict["total_bpkt"].append(bpkt_total)

    processed_df_dict["total_fbyt"].append(fbyt_total)
    processed_df_dict["total_bbyt"].append(bbyt_total)

    processed_df_dict["min_fbyt"].append(fbyt_min)
    processed_df_dict["min_bbyt"].append(bbyt_min)

    processed_df_dict["max_fbyt"].append(fbyt_max)
    processed_df_dict["max_bbyt"].append(bbyt_max)

    processed_df_dict["mean_fbyt"].append(fbyt_mean)
    processed_df_dict["mean_bbyt"].append(bbyt_mean)

    processed_df_dict["std_fbyt"].append(fbyt_std)
    processed_df_dict["std_bbyt"].append(bbyt_std)

    # print(processed_df_dict)

    return pd.DataFrame(processed_df_dict)