import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, DecisionTreeAttack
from sklearn.metrics import log_loss
import pickle
import random

from utils import *

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
args = parser.parse_args()
result_dir = args.folder


asset_model_dir = os.path.join("../../", "asset", "pants-app-end-host", "vanilla")
asset_common_dir = os.path.join("../../", "asset", "pants-app-end-host", "common")
asset_transferability_dir = os.path.join("../../", "asset", "transferability")

scaler = joblib.load(os.path.join(asset_model_dir, "handcraft_scaler.save"))

clf = RandomForestClassifier(max_depth=12, random_state=0)
clf = pickle.load(open(os.path.join(asset_model_dir, "rf.pkl"), "rb"))


with np.load(os.path.join(asset_common_dir, "data_handcraft.npy")) as data:
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

idx_train = X_train[:, 0].astype(int)
idx_test = X_test[:, 0].astype(int)

# files = os.listdir(adv_sample_folder)
correct_sample_idx_set = set()

for sample_idx in idx_test:
    gt = Y_test[np.where(idx_test == int(sample_idx))[0]][0]
    feature = X_test[np.where(idx_test == int(sample_idx))[0], 1:]
    norm_return_df = data_normalization(feature, scaler)
    reconstructed_adv_inputs = torch.from_numpy(
        norm_return_df[0, :].astype(np.float32)
    )
    feature = reconstructed_adv_inputs.reshape(1, -1)

    adv_pred = clf.predict(feature)
    if adv_pred == gt:
        correct_sample_idx_set.add(int(sample_idx))

correct_sample_dict = {}
for sample_idx in correct_sample_idx_set:
    correct_sample_dict[sample_idx] = 0

visited = {}

for t in range(1):
    if t == 0:
        folder = os.path.join(asset_transferability_dir, "cnn/adv_samples")
    if t == 1:
        folder = os.path.join(asset_transferability_dir, "cnn/unsuccess_samples")
        
    files = os.listdir(folder)
    idx_set = set([int(file.split(".")[0].split("_")[1]) for file in files])
    idx_set = idx_set.intersection(correct_sample_idx_set)
    sampled_idx_set = random.sample(list(idx_set), 30)
    
    for file in files:
        sample_idx = int(file.split(".")[0].split("_")[1])
        if sample_idx not in sampled_idx_set:
            continue
        if sample_idx not in correct_sample_idx_set:
            continue
        if correct_sample_dict[sample_idx] == 1:
            continue
        path = os.path.join(folder, file)

        bi_flow = pd.read_csv(path)
        orig_flow = pd.read_csv(os.path.join(asset_common_dir, "pkt_dir", f"{sample_idx}.csv"))
        orig_pkt_len = len(orig_flow)
        if orig_pkt_len <= 400:
            num_untouched_pkts = count_consecutive_zeros_at_end(list(bi_flow["pkt_len"]))
            bi_flow = bi_flow.head(400 - num_untouched_pkts)
            bi_flow = np.array(bi_flow)
        else:
            remaining_flow = cast_to_tf_format(orig_flow)[400:, :]
            bi_flow = np.array(bi_flow)
            bi_flow = np.concatenate([bi_flow, remaining_flow], axis=0)
        bi_flow = pd.DataFrame(bi_flow, columns=["pkt_len", "iat"])
        bi_flow["time"] = bi_flow["iat"].cumsum()
        bi_flow["fwd"] = bi_flow['pkt_len'].apply(lambda x: True if x > 0 else False)
        feature = tf_format_to_feature(bi_flow)
        norm_return_df = data_normalization(feature, scaler)
        reconstructed_adv_inputs = torch.from_numpy(
            norm_return_df[0, :].astype(np.float32)
        )
        adv_feature = reconstructed_adv_inputs.reshape(1, -1)
        adv_pred = clf.predict(adv_feature)
        gt = Y_test[np.where(idx_test == int(sample_idx))[0]][0]
        if adv_pred != gt and correct_sample_dict[sample_idx] == 0:
            correct_sample_dict[sample_idx] = 1

        if sample_idx not in visited:
            visited[sample_idx] = 0
        if adv_pred != gt and visited[sample_idx] == 0:
            visited[sample_idx] = 1
        

l = f"Transferability cnn -> rf: {sum(visited.values()) / len(visited.values())}"
print(l)

with open(os.path.join(args.folder, "cnn_to_rf.txt"), "w") as file:
    file.write(l)


