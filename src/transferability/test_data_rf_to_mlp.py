import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, AdamW
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

adv_sample_folders = [
    os.path.join(asset_transferability_dir, "rf/adv_samples"),
    # os.path.join(asset_transferability_dir, "rf/unsuccess_samples")
    ]

scaler = joblib.load(os.path.join(asset_model_dir, "handcraft_scaler.save"))

input_dim = 22
num_layers = 4
num_units = 200
output_dim = 14
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        layers = []
        layer_num_units = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(layer_num_units, num_units))
            layers.append(torch.nn.ReLU())
            layer_num_units = num_units
        layers.append(torch.nn.Linear(layer_num_units, output_dim))

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        return x
mlp_classifier = Network()
mlp_classifier.load_state_dict(torch.load(os.path.join(asset_model_dir, "mlp_handcraft.pt")))
mlp_classifier.eval()

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
    gt = Y_test[int(np.where(idx_test == int(sample_idx))[0][0])]
    feature = X_test[np.where(idx_test == int(sample_idx))[0], 1:]
    norm_return_df = data_normalization(feature, scaler)
    reconstructed_adv_inputs = torch.from_numpy(
        norm_return_df[0, :].astype(np.float32)
    )
    feature = reconstructed_adv_inputs.reshape(1, -1)

    adv_pred = mlp_classifier(feature)
    adv_pred = torch.argmax(adv_pred, dim=1).item()
    if adv_pred == gt:
        correct_sample_idx_set.add(int(sample_idx))


correct_sample_dict = {}
for sample_idx in correct_sample_idx_set:
    correct_sample_dict[sample_idx] = 0

visited = {}
for folder in adv_sample_folders:
    
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
        feature = transfer_to_features(bi_flow)

        norm_return_df = data_normalization(feature, scaler)
        reconstructed_adv_inputs = torch.from_numpy(
            norm_return_df[0, :].astype(np.float32)
        )
        adv_feature = reconstructed_adv_inputs.reshape(1, -1)
        adv_pred = mlp_classifier(adv_feature)
        adv_pred = torch.argmax(adv_pred, dim=1).item()
        gt = Y_test[int(np.where(idx_test == int(sample_idx))[0][0])]
        if adv_pred != gt and correct_sample_dict[sample_idx] == 0:
            correct_sample_dict[sample_idx] = 1

        if sample_idx not in visited:
            visited[sample_idx] = 0
        if adv_pred != gt and visited[sample_idx] == 0:
            visited[sample_idx] = 1
        

l = f"Transferability rf -> mlp: {sum(visited.values()) / len(visited.values())}"
print(l)

with open(os.path.join(args.folder, "rf_to_mlp.txt"), "w") as file:
    file.write(l)

