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


first_n_pkts = 400

asset_model_dir = os.path.join("../../", "asset", "pants-app-end-host", "vanilla")
asset_common_dir = os.path.join("../../", "asset", "pants-app-end-host", "common")
asset_transferability_dir = os.path.join("../../", "asset", "transferability")

five_tuple = ["srcip", "dstip", "srcport", "dstport", "proto"]

# def handle_flow(flow):
#     first_n_pkts = 400
#     flow = flow.head(first_n_pkts)
#     flow.sort_values(by="time", inplace=True)
#     flow_ndarray = []
#     fwd_five_tuple = list(flow.iloc[0][five_tuple])
#     bwd_five_tuple = [
#         fwd_five_tuple[1],
#         fwd_five_tuple[0],
#         fwd_five_tuple[3],
#         fwd_five_tuple[2],
#         fwd_five_tuple[4],
#     ]

#     iat = np.diff(flow["time"])
#     iat = np.insert(iat, 0, 0)

#     for index, row in flow.iterrows():
#         if list(row[five_tuple]) == fwd_five_tuple:
#             flow_ndarray.append([row["pkt_len"], iat[index]])
#         elif list(row[five_tuple]) == bwd_five_tuple:
#             flow_ndarray.append([-row["pkt_len"], iat[index]])
#         else:
#             print("Error")
        
    
#     remaining_pkts = first_n_pkts - len(flow_ndarray) if len(flow_ndarray) < first_n_pkts else 0

#     if remaining_pkts > 0:
#         for _ in range(remaining_pkts):
#             flow_ndarray.append([0, 0])
    
#     flow_ndarray = np.array(flow_ndarray)
#     out = list(flow_ndarray[:, 0]) + list(flow_ndarray[:, 1])
#     return np.array(out).astype(float) 

class DF(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DF, self).__init__()
        self.block1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=input_shape,              
                out_channels=32,            
                kernel_size=8,              
                stride=1,                   
                padding="same",                 
            ),  
            nn.BatchNorm1d(32),                   
            nn.ELU(alpha=1.0),                     
            nn.Conv1d(32, 32, 8, 1, "same"),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(8, 4, 0), 
            nn.Dropout(0.1), 
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, 1, "same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, 1, "same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 8, 1, "same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, 1, "same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 8, 1, "same"),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, 1, "same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.fc1 = nn.Sequential(         
            nn.Flatten(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),              
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),              
        )

        self.out = nn.Sequential(
            nn.Linear(512, num_classes),
        )   

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc1(x)
        x = self.fc2(x)          
        output = self.out(x)
        return output

# Load the model
model = DF(input_shape=1, num_classes=14)
model.load_state_dict(torch.load(os.path.join(asset_model_dir, 'cnn.pth'), map_location=torch.device('cpu')))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

with np.load(os.path.join(asset_common_dir, "cnn_data.npy"), allow_pickle=True) as data:
    X_train = data["X_train"].astype(float) 
    Y_train = data["Y_train"]
    X_test = data["X_test"].astype(float)
    Y_test = data["Y_test"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]

X_train[:, first_n_pkts:] = np.log1p(X_train[:, first_n_pkts:])
X_test[:, first_n_pkts:] = np.log1p(X_test[:, first_n_pkts:])
X_train[:, :first_n_pkts] = np.log1p(np.abs(X_train[:, :first_n_pkts])) * np.sign(X_train[:, :first_n_pkts])
X_test[:, :first_n_pkts] = np.log1p(np.abs(X_test[:, :first_n_pkts])) * np.sign(X_test[:, :first_n_pkts])
# Normalize data
all_iats = X_train[:, first_n_pkts:]
all_pkt_lens = X_train[:, :first_n_pkts]
max_iats = np.max(all_iats)
min_iats = np.min(all_iats)
max_pkt_lens = np.max(all_pkt_lens)
min_pkt_lens = np.min(all_pkt_lens)

X_train[:, first_n_pkts:] = (X_train[:, first_n_pkts:] - min_iats) / (max_iats - min_iats)
X_test[:, first_n_pkts:] = (X_test[:, first_n_pkts:] - min_iats) / (max_iats - min_iats)
X_train[:, :first_n_pkts] = (X_train[:, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
X_test[:, :first_n_pkts] = (X_test[:, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)

X_train = X_train[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]

# files = []
# for folder in adv_sample_folders:
#     files += os.listdir(folder)
correct_sample_idx_set = set()

# for sample_idx in idx_test:
#     gt = Y_test[np.where(idx_test == int(sample_idx))[0]]
#     feature = X_test[np.where(idx_test == int(sample_idx))[0], 1:]
#     norm_return_df = data_normalization(feature, scaler)
#     reconstructed_adv_inputs = torch.from_numpy(
#         norm_return_df[0, :].astype(np.float32)
#     )
#     feature = reconstructed_adv_inputs.reshape(1, -1)

#     adv_pred = clf.predict(feature)
#     if adv_pred == gt:
#         correct_sample_idx_set.add(int(sample_idx))

# print(len(correct_sample_idx_set) / len(X_test))

for sample_idx in idx_test:
    gt = Y_test[np.where(idx_test == int(sample_idx))[0]]
    feature = X_test[np.where(idx_test == int(sample_idx))[0]]
    # norm_return_df = data_normalization(feature, scaler)
    # reconstructed_adv_inputs = torch.from_numpy(
    #     norm_return_df[0, :].astype(np.float32)
    # )
    # feature = reconstructed_adv_inputs.reshape(1, -1)
    adv_pred = model(torch.tensor(feature, dtype=torch.float32))
    adv_pred = torch.argmax(adv_pred, dim=1).item()
    if adv_pred == gt[0]:
        correct_sample_idx_set.add(int(sample_idx))

correct_sample_dict = {}
for sample_idx in correct_sample_idx_set:
    correct_sample_dict[sample_idx] = 0

i = 0
visited = {}
for t in range(1):
    if t == 0:
        folder = os.path.join(asset_transferability_dir, "tf/adv_samples")
    if t == 1:
        folder = os.path.join(asset_transferability_dir, "tf/unsuccess_samples")

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

        flow = pd.read_csv(path)
        flow = np.array(flow)
        pkt_lens = flow[:, 0]
        iats = flow[:, 1]

        flow = np.concatenate([pkt_lens, iats], axis=0)
        flow = flow[np.newaxis, :]

        flow[:, first_n_pkts:] = np.log1p(flow[:, first_n_pkts:])
        flow[:, :first_n_pkts] = np.log1p(np.abs(flow[:, :first_n_pkts])) * np.sign(flow[:, :first_n_pkts])
        flow[:, first_n_pkts:] = (flow[:, first_n_pkts:] - min_iats) / (max_iats - min_iats)
        flow[:, :first_n_pkts] = (flow[:, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)

        flow = flow[:, np.newaxis, :]

        flow = torch.tensor(flow, dtype=torch.float32).to(device)
        gt = Y_test[np.where(idx_test == int(sample_idx))[0]]
        
        adv_pred = model(flow)
        adv_pred = torch.argmax(adv_pred, dim=1).item()
        if adv_pred != gt[0] and correct_sample_dict[sample_idx] == 0:
            correct_sample_dict[sample_idx] = 1

        if sample_idx not in visited:
            visited[sample_idx] = 0
        if adv_pred != gt and visited[sample_idx] == 0:
            visited[sample_idx] = 1
        
l = f"Transferability tf -> cnn: {sum(visited.values()) / len(visited.values())}"
print(l)

with open(os.path.join(args.folder, "tf_to_cnn.txt"), "w") as file:
    file.write(l)

