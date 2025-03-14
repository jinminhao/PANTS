import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils_cnn import *
import pandas as pd
import argparse
import time
import os

first_n_pkts = 400
max_overhead = 0.2

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-d", "--asset_name", type=str, help="Name of dir for the asset")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
parser.add_argument("-a", "--attack", type=str, help="attack type")
parser.add_argument("-s", "--slice", help="slice", type=int, default=0, required=False)
parser.add_argument(
    "-n",
    "--num",
    help="number of samples evaluated (only for attacking test set)",
    type=int,
    default=-1,
    required=False,
)

# Parse the arguments
args = parser.parse_args()
asset_model_dir = os.path.join("../../../", "asset", "pants-vpn-end-host", args.asset_name)
asset_common_dir = os.path.join("../../../", "asset", "pants-vpn-end-host", "common")
if not os.path.isdir(asset_model_dir):
    print(f"The asset dir: {asset_model_dir} doesn't exist.")
    exit()

attack_train = None
if args.attack == "train":
    attack_train = True
elif args.attack == "test":
    attack_train = False
else:
    print("Invalid arg for attack. Please use either train or test.")
    exit()

if attack_train == True:
    slice_id = args.slice
    print(f"working on {slice_id} piece")
    print("Maximum 3 piece")
    if slice_id < 0 or slice_id > 2:
        print("Wrong slice id")
        exit()

result_folder = args.folder
if attack_train == True:
    result_folder = f"{result_folder}_piece_{slice_id}"
    if "aug" not in result_folder:
        print("should have aug in the result folder name")
        exit()
if os.path.exists(result_folder):
    print("Folder already exists")
    exit()

os.mkdir(result_folder)
os.mkdir(os.path.join(result_folder, "adv_samples"))
os.mkdir(os.path.join(result_folder, "unsuccess_samples"))

path = result_folder

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
with np.load(
    os.path.join(asset_model_dir, "normalization_cnn.npy"), allow_pickle=True
) as data:
    max_iats = data["max_iats"]
    min_iats = data["min_iats"]
    max_pkt_lens = data["max_pkt_lens"]
    min_pkt_lens = data["min_pkt_lens"]

X_train[:, first_n_pkts:] = (X_train[:, first_n_pkts:] - min_iats) / (max_iats - min_iats)
X_test[:, first_n_pkts:] = (X_test[:, first_n_pkts:] - min_iats) / (max_iats - min_iats)
X_train[:, :first_n_pkts] = (X_train[:, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
X_test[:, :first_n_pkts] = (X_test[:, :first_n_pkts] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)

X_train = X_train[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]

normalized_direction_value = (np.log1p(0) - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
normalized_max_pkt_len = (np.log1p(1500) - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
normalized_min_pkt_len = (np.log1p(40) - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)

# Preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, sample_idx):
        self.data = data
        self.labels = labels
        self.sample_idx = sample_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            "index": torch.tensor(self.sample_idx[idx], dtype=torch.long),
        }

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

model = DF(input_shape=1, num_classes=2)
model.load_state_dict(
    torch.load(os.path.join(asset_model_dir, "cnn.pth"), map_location=torch.device("cpu"))
)
model.eval()
# PANTS is heavily relying on CPU as it is iteratively interacting with an SMT solver. 
# device is forced to be "cpu" as cuda will not speed up the process.
device = "cpu"
model.to(device)


# Create test dataset and dataloader
if attack_train == False:
    dataset = TimeSeriesDataset(X_test, Y_test, idx_test)
    if args.num == -1:
        num_evaluated_samples = len(X_test)
    else:
        num_evaluated_samples = args.num
elif attack_train == True:
    dataset = TimeSeriesDataset(X_train, Y_train, idx_train)
    num_evaluated_samples = len(X_train)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

correct_X = []
correct_Y = []
correct_idx = []
pred_list = []
label_list = []
with torch.no_grad():
    correct = 0
    total = 0
    for batch in dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        sample_idx = batch['index'].to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if predicted.item() == labels.item():
            correct_X.append(inputs)
            correct_Y.append(labels)
            correct_idx.append(sample_idx)
        pred_list.append(predicted.item())
        label_list.append(labels.item())

    accuracy = correct / total
    f1 = f1_score(label_list, pred_list, average="weighted")
    print(f'Accuracy: {accuracy}, f1: {f1}')

slice_data_volumn = np.ceil(len(correct_X) / 3)
success = 0
total = 0
total_num_adv_samples = 0
total_generated_outputs = 0
smt_cross_entropy_results = []
time_start = time.time()
for idx in range(len(correct_X)):
    if attack_train and slice_id == 0:
        if idx >= slice_data_volumn:
            break

    if attack_train and slice_id == 1:
        if idx < slice_data_volumn:
            continue
        if idx >= 2 * slice_data_volumn:
            break

    if attack_train and slice_id == 2:
        if idx < 2 * slice_data_volumn:
            continue
    #     if idx > 3 * slice_data_volumn:
    #         break

    # if attack_train and slice_id == 3:
    #     if idx < 3 * slice_data_volumn:
    #         continue
    #     if idx > 4 * slice_data_volumn:
    #         break
    
    # if attack_train and slice_id == 4:
    #     if idx < 4 * slice_data_volumn:
    #         continue
    #     if idx > 5 * slice_data_volumn:
    #         break

    # if attack_train and slice_id == 5:
    #     if idx < 5 * slice_data_volumn:
    #         continue
    #     if idx > 6 * slice_data_volumn:
    #         break

    # if attack_train and slice_id == 6:
    #     if idx < 6 * slice_data_volumn:
    #         continue
    #     if idx > 7 * slice_data_volumn:
    #         break

    # if attack_train and slice_id == 7:
    #     if idx < 7 * slice_data_volumn:
    #         continue
    #     if idx > 8 * slice_data_volumn:
    #         break
    
    # if attack_train and slice_id == 8:
    #     if idx < 8 * slice_data_volumn:
    #         continue
    inputs = correct_X[idx]
    labels = correct_Y[idx]
    sample_idx = correct_idx[idx]

    original_thorough_flow = pd.read_csv(
        os.path.join(asset_common_dir, "pkt_dir", f"{sample_idx.item()}.csv")
    )
    original_thorough_iat = np.diff(original_thorough_flow["time"])
    original_thorough_iat = np.insert(original_thorough_iat, 0, 0)
    orig_flow_len = len(original_thorough_flow)
    if orig_flow_len > first_n_pkts:
        orig_flow_len = first_n_pkts

    if len(original_thorough_flow) > first_n_pkts:
        remaining_sum_iats = sum(original_thorough_iat[first_n_pkts:])
    else:
        remaining_sum_iats = 0

    success_indicator = False
    pgd_adv_features_list = get_pgd_adv_features(inputs,
                        labels,
                        orig_flow_len,
                        model,
                        nn.CrossEntropyLoss(), 
                        5, 
                        20, 
                        0.005, 
                        normalized_direction_value, 
                        normalized_max_pkt_len,
                        normalized_min_pkt_len)

    final_pgd_adv_features_list = []
    smt_process_list = []
    for n, adv_inputs_set in enumerate(pgd_adv_features_list):
        adv_inputs = adv_inputs_set[0]
        adv_inputs_len = adv_inputs_set[1]
        injected_pkt_idx = adv_inputs_set[2]
        denorm_adv_inputs = adv_inputs.cpu().clone().detach()
        denorm_adv_inputs = denormalize(denorm_adv_inputs, max_iats, min_iats, max_pkt_lens, min_pkt_lens)

        for i in range(first_n_pkts):
            if denorm_adv_inputs[0, 0, i] > 1500:
                denorm_adv_inputs[0, 0, i] = 1500
            elif denorm_adv_inputs[0, 0, i] < 40 and denorm_adv_inputs[0, 0, i] > 0:
                denorm_adv_inputs[0, 0, i] = 40

        perturbable_mask = get_perturbable_mask(adv_inputs, adv_inputs_len)
        denorm_adv_inputs = denorm_adv_inputs * perturbable_mask
        tmp_denorm_adv_inputs = np.array([denorm_adv_inputs[0, 0, :first_n_pkts].numpy(), 
                                          denorm_adv_inputs[0, 0, first_n_pkts:].numpy()])
        tmp_denorm_adv_inputs = np.transpose(tmp_denorm_adv_inputs)
        df_denorm_adv_inputs = pd.DataFrame(tmp_denorm_adv_inputs, columns=["pkt_len", "iat"])
        # .to_csv(f"{path}/adv_samples/adv_{sample_idx.item()}_{n}.csv", index=False)

        denorm_inputs = inputs.cpu().clone().detach()
        denorm_inputs = denormalize(denorm_inputs, max_iats, min_iats, max_pkt_lens, min_pkt_lens)
        tmp_denorm_inputs = np.array([denorm_inputs[0, 0, :first_n_pkts].numpy(),
                                        denorm_inputs[0, 0, first_n_pkts:].numpy()])
        tmp_denorm_inputs = np.transpose(tmp_denorm_inputs)
        df_denorm_inputs = pd.DataFrame(tmp_denorm_inputs, columns=["pkt_len", "iat"])

        df_denorm_ref_inputs = df_denorm_inputs.copy()
        for idx in injected_pkt_idx:
            row_to_insert = df_denorm_adv_inputs.iloc[idx]
            row_to_insert["iat"] = 1
            df_denorm_ref_inputs = pd.concat(
                [df_denorm_ref_inputs.iloc[:idx], pd.DataFrame([row_to_insert]), df_denorm_ref_inputs.iloc[idx:]]
            ).reset_index(drop=True)
        df_denorm_ref_inputs = df_denorm_ref_inputs.iloc[:first_n_pkts]
        adv_iats = df_denorm_adv_inputs["iat"]
        orig_iats = df_denorm_inputs["iat"]

        overhead = (sum(adv_iats) - sum(orig_iats)) / (sum(orig_iats) + remaining_sum_iats)
        if overhead > max_overhead:
            sorted_idx = get_important_features(df_denorm_adv_inputs, df_denorm_ref_inputs)
            smt_process_list.append((df_denorm_adv_inputs, df_denorm_ref_inputs, max_overhead, sorted_idx, remaining_sum_iats))
        else:
            final_pgd_adv_features_list.append(df_denorm_adv_inputs)
    if len(smt_process_list) > 0:
        df_smt_out = call_smt(smt_process_list)
        final_pgd_adv_features_list += df_smt_out

    for adv_feat in final_pgd_adv_features_list:
        adv_feat = np.array(list(adv_feat["pkt_len"]) + list(adv_feat["iat"]))
        adv_feat = np.expand_dims(adv_feat, axis=0)
        adv_feat = np.expand_dims(adv_feat, axis=0)

    # adv_inputs[:, :, 1] = np.log1p(adv_inputs[:, :, 1])
    # adv_inputs[:, :, 0] = np.log1p(np.abs(adv_inputs[:, :, 0])) * np.sign(adv_inputs[:, :, 0])
    # adv_inputs[:, :, 1] = (adv_inputs[:, :, 1] - min_iats) / (max_iats - min_iats)
    # adv_inputs[:, :, 0] = (adv_inputs[:, :, 0] - min_pkt_lens) / (max_pkt_lens - min_pkt_lens)
        adv_inputs = normalize(adv_feat, max_iats, min_iats, max_pkt_lens, min_pkt_lens)
        adv_inputs = torch.from_numpy(adv_inputs).float().to(device)
    
        adv_outputs = model(adv_inputs)
        smt_cross_entropy_results.append(nn.CrossEntropyLoss()(adv_outputs, labels).item())
        if torch.max(adv_outputs, 1)[1].item() != labels.item() and success_indicator == False:
            success += 1
            success_indicator = True
            break
    total_generated_outputs += len(final_pgd_adv_features_list)
    if success_indicator:
        total_num_adv_samples += len(final_pgd_adv_features_list)
        # if attack_train:
        for adv_flow_idx, final_adv_flow in enumerate(final_pgd_adv_features_list):
            # TODO!!
            final_adv_flow.to_csv(
                os.path.join(
                    result_folder,
                    "adv_samples",
                    f"adv_{int(sample_idx.item())}_{adv_flow_idx}.csv",
                ),
                index=False,
                )
    else:
        if attack_train:
            for adv_flow_idx, final_adv_flow in enumerate(final_pgd_adv_features_list):
                # TODO!!
                final_adv_flow.to_csv(
                    os.path.join(
                        result_folder,
                        "unsuccess_samples",
                        f"{int(sample_idx.item())}_{adv_flow_idx}.csv",
                    ),
                    index=False,
                    )
    total += 1
    
    print(f"Progress: {total} / {num_evaluated_samples}")
    if total % 10 == 0:
        now = time.time()
        time_elapsed = now - time_start
        if total_num_adv_samples != 0:
            l3 = f"samples attacked: {total}, ASR: {success / total}, speed: {total_num_adv_samples/time_elapsed}\n"
        else:
            l3 = f"samples attacked: {total}, ASR: {success / total}, speed: {total_num_adv_samples/time_elapsed}\n"
        print(l3)
        asr_path = os.path.join(result_folder, "result.txt")
        file = open(asr_path, "a")
        file.write(l3)
        file.close()

    if attack_train == False and total == num_evaluated_samples:
        break

now = time.time()
time_elapsed = now - time_start
l3 = f"Summary: ASR: {success / total}, speed: {total_num_adv_samples/time_elapsed}\n"
print(l3)
asr_path = os.path.join(result_folder, "result.txt")
file = open(asr_path, "a")
file.write(l3)
file.close()