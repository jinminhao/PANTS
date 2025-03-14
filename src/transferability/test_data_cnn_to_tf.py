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

first_n_pkts=400


asset_model_dir = os.path.join("../../", "asset", "pants-app-end-host", "vanilla")
asset_common_dir = os.path.join("../../", "asset", "pants-app-end-host", "common")
asset_transferability_dir = os.path.join("../../", "asset", "transferability")

five_tuple = ["srcip", "dstip", "srcport", "dstport", "proto"]

def handle_flow(flow):
    first_n_pkts = 400
    flow = flow.head(first_n_pkts)
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
        
    
    remaining_pkts = first_n_pkts - len(flow_ndarray) if len(flow_ndarray) < first_n_pkts else 0

    if remaining_pkts > 0:
        for _ in range(remaining_pkts):
            flow_ndarray.append([0, 0])
    return np.array(flow_ndarray).astype(float) 

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TimeSeriesTransformer, self).__init__()
        config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=256,
            max_position_embeddings=5000,
        )
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = BertModel(config)
        self.fc = nn.Linear(128 * 400, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Project input to hidden_dim
        attention_mask = (x != 0).sum(dim=-1) > 0  # Create attention mask
        outputs = self.transformer(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, :, :]  # Take [CLS] token output
        logits = self.fc(cls_output.view(-1, 128 * 400))
        return logits

# Load the model
model = TimeSeriesTransformer(input_dim=2, hidden_dim=128, num_classes=14)
model.load_state_dict(torch.load(os.path.join(asset_model_dir, 'transformer.pth'), map_location=torch.device('cpu')))
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

# files = os.listdir(adv_sample_folder)
correct_sample_idx_set = set()

for sample_idx in idx_test:
    gt = Y_test[np.where(idx_test == int(sample_idx))[0]]
    feature = X_test[np.where(idx_test == int(sample_idx))[0]]
    pkt_lens = feature[0, 0, :400].reshape(400, 1)
    iats = feature[0, 0, 400:].reshape(400, 1)
    flow = np.concatenate([pkt_lens, iats], axis=1).reshape(1, 400, 2)

    adv_pred = model(torch.tensor(flow, dtype=torch.float32).to(device))
    adv_pred = torch.argmax(adv_pred, dim=1).item()
    if adv_pred == gt[0]:
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
        if sample_idx not in correct_sample_idx_set:
            continue
        if correct_sample_dict[sample_idx] == 1:
            continue
        path = os.path.join(folder, file)

        flow = pd.read_csv(path)
        flow = np.array(flow)
        flow = flow.reshape(-1, flow.shape[0], flow.shape[1])

        flow[:, :, 1] = np.log1p(flow[:, :, 1])
        flow[:, :, 0] = np.log1p(np.abs(flow[:, :, 0])) * np.sign(flow[:, :, 0])
        flow[:, :, 1] = (flow[:, :, 1] - min_iats) / (max_iats - min_iats)
        flow[:, :, 0] = (flow[:, :, 0] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)

        adv_pred = model(torch.tensor(flow, dtype=torch.float32).to(device))
        adv_pred = torch.argmax(adv_pred, dim=1).item()
        if adv_pred != gt and correct_sample_dict[sample_idx] == 0:
            correct_sample_dict[sample_idx] = 1

        if sample_idx not in visited:
            visited[sample_idx] = 0
        if adv_pred != gt and visited[sample_idx] == 0:
            visited[sample_idx] = 1
    

l = f"Transferability cnn -> tf: {sum(visited.values()) / len(visited.values())}"
print(l)

with open(os.path.join(args.folder, "cnn_to_tf.txt"), "w") as file:
    file.write(l)


