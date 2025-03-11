import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib
from util import *

import shap
import matplotlib.pyplot as plt

from smt import call_smt
import multiprocessing as mp
import time
import os
import json

import warnings
import argparse

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy("file_system")


def get_pgd_adv_features(orig_feature, model, loss, total_repeat_time, iters, alpha):

    pgd_adv_features_list = []
    perturbed_feature_ = orig_feature.clone().detach()

    for _ in range(total_repeat_time):
        perturbed_feature = perturbed_feature_ + torch.zeros_like(
            perturbed_feature_
        ).normal_(mean=0, std=1e-1)
        for _ in range(iters):
            perturbed_feature.requires_grad = True
            outputs = model(perturbed_feature)
            model.zero_grad()
            cost = loss(outputs, labels)
            # - 1e-4 * reg_loss
            cost.backward()

            sign_grad = perturbed_feature.grad.sign()
            # Determine if the sign of biat part is less than 0. If true, mask to 0
            mask = mask_feat(sign_grad)

            perturbed_feature = perturbed_feature + alpha * sign_grad
            # * mask

            perturbed_feature = torch.clamp(perturbed_feature, min=0, max=1).detach_()

        added_features = perturbed_feature.clone().detach()
        pgd_adv_features_list.append(added_features)

    return pgd_adv_features_list


def smt_solver(
    pgd_perturbed_feature,
    index,
    orig_feature,
    mode,
    considerred_important_features,
    considerred_important_feature_indices,
    only_src,
):
    # SMT
    denomalized_adv_inputs = data_denormalization(pgd_perturbed_feature, scaler)
    denormalized_orig_inputs = data_denormalization(orig_feature, scaler)
    # append index
    denomalized_adv_inputs = np.concatenate(([index], denomalized_adv_inputs[0]))
    denormalized_orig_inputs = np.concatenate(([index], denormalized_orig_inputs[0]))
    column_names = ["index"] + list(dataset_fields.keys())
    denomalized_adv_input = pd.DataFrame([denomalized_adv_inputs], columns=column_names)
    denormalized_orig_inputs = pd.DataFrame(
        [denormalized_orig_inputs], columns=column_names
    )

    concat = pd.concat([denomalized_adv_input, denormalized_orig_inputs]).reset_index(
        drop=True
    )
    results = call_smt(
        mode,
        concat,
        considerred_important_features,
        considerred_important_feature_indices,
        only_src,
        asset_common_dir,
        tolerance=0.2,
    )

    return results


def get_feature_and_solve(
    mode,
    important_feature_list,
    important_feature_indices_list,
    pgd_perturbed_feature,
    index,
    orig_feature,
    only_src,
):
    smt_results = []
    # First try with mode = "single"
    # mode = "single"
    considerred_important_features = []
    considerred_important_feature_indices = []
    if mode == "single":
        max_num_important_features = 6
    else:
        max_num_important_features = 3
    for num_important_features in range(1, max_num_important_features):
        considerred_important_features.append(
            important_feature_list[num_important_features - 1]
        )
        considerred_important_feature_indices.append(
            important_feature_indices_list[num_important_features - 1]
        )

        results = smt_solver(
            pgd_perturbed_feature=pgd_perturbed_feature,
            index=index,
            orig_feature=orig_feature,
            mode=mode,
            considerred_important_features=considerred_important_features,
            considerred_important_feature_indices=considerred_important_feature_indices,
            only_src=only_src,
        )
        smt_results += results
        if len(considerred_important_features) == 3:
            break
        if not all([x[0] for x in results]) or len(results) == 0:
            considerred_important_features = considerred_important_features[:-1]
            considerred_important_feature_indices = (
                considerred_important_feature_indices[:-1]
            )

    return smt_results


def pgd_attack_RANDOM_INIT_v3(
    model,
    index,
    features,
    labels,
    only_src,
    eps=0.1,
    alpha=0.01,
    iters=20,
    num_restarts=1,
):
    attack_success = False
    candidate_features = []
    pgd_adv_feature = None
    attack_success = False
    orig_feature = features.data
    loss = torch.nn.CrossEntropyLoss()

    denorm_orig_features = data_denormalization(orig_feature, scaler)
    denorm_orig_features = pd.DataFrame(
        denorm_orig_features, columns=list(dataset_fields.keys())
    )
    orig_pkts = int(np.round(denorm_orig_features["l_num_pkts"]))

    loss_list = []
    adv_flow_list = []
    considered_important_feature_superlist = []
    for _ in range(num_restarts):
        # The std is set to 1e-7 to avoid the perturbed features to be the same as the original one
        # Value should be super small
        eps_init = torch.zeros_like(features).normal_(mean=0, std=1e-7)
        perturbed_feature = orig_feature.clone().detach() + eps_init

        total_repeat_time = 5
        pgd_adv_features_list = get_pgd_adv_features(
            orig_feature=perturbed_feature,
            model=model,
            loss=loss,
            total_repeat_time=total_repeat_time,
            iters=iters,
            alpha=alpha,
        )

        final_results = []
        for pgd_perturbed_feature in pgd_adv_features_list:

            # Try to figure out what features are important and smt
            (
                important_feature_list,
                important_feature_indices_list,
            ) = get_important_features_v2(
                pgd_perturbed_feature, orig_feature.detach().numpy()
            )

            # pgd_perturbed_feature = self_correct_pkt_num(
            #     perturbed_features=pgd_perturbed_feature,
            #     scaler=scaler,
            #     important_feature_list=important_feature_list,
            #     important_feature_indices_list=important_feature_indices_list,
            #     orig_feature=denorm_orig_features,
            # )

            results = get_feature_and_solve(
                mode="single",
                important_feature_list=important_feature_list,
                important_feature_indices_list=important_feature_indices_list,
                pgd_perturbed_feature=pgd_perturbed_feature,
                index=index,
                orig_feature=orig_feature,
                only_src=only_src,
            )
            if_success = [x[0] for x in results if x[0] == True]
            if len(if_success) == 0:
                results = get_feature_and_solve(
                    mode="chunks",
                    important_feature_list=important_feature_list,
                    important_feature_indices_list=important_feature_indices_list,
                    pgd_perturbed_feature=pgd_perturbed_feature,
                    index=index,
                    orig_feature=orig_feature,
                    only_src=only_src,
                )

            final_results += results

        for result in final_results:
            smt_success, smt_feature, adv_flow, considered_important_feature_list = (
                result
            )
            if smt_success:
                norm_return_df = data_normalization(smt_feature, scaler)
                reconstructed_adv_inputs = torch.from_numpy(
                    norm_return_df[0, :].astype(np.float32)
                )
                adv_feature = reconstructed_adv_inputs.reshape(1, -1)
                adv_pred = model(adv_feature)
                if attack_success == False and torch.argmax(adv_pred).item() != labels:
                    attack_success = True
                loss_list.append(loss(adv_pred, labels).detach().item())
                adv_flow_list.append(adv_flow)
                considered_important_feature_superlist.append(
                    considered_important_feature_list
                )
        pred = model(features)
        orig_loss = loss(pred, labels).detach().item()

        if len(loss_list) != 0:
            break
    if len(loss_list) == 0:
        adv_loss = orig_loss
        # final_adv_flow = None
        adv_flow_list = []
        considered_important_feature_superlist = []
    else:
        adv_loss = max(loss_list)
        # final_adv_flow = adv_flow_list[np.argmax(loss_list)]
    return (
        adv_loss,
        orig_loss,
        attack_success,
        adv_flow_list,
        considered_important_feature_superlist,
    )


parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
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
asset_model_dir = os.path.join("../../../", "asset", "pants-vca-end-host", args.asset_name)
asset_common_dir = os.path.join("../../../", "asset", "pants-vca-end-host", "common")
if not os.path.isdir(asset_model_dir):
    print(f"The asset dir: {asset_model_dir} doesn't exist.")
    exit()

only_src = True

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

batch_size = 1

with np.load(os.path.join(asset_common_dir, "data_handcraft.npy")) as data:
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

if attack_train:
    X_test_orig = X_train
    num_evaluated_samples = len(X_train)
else:
    X_test_orig = X_test
    if args.num == -1:
        num_evaluated_samples = len(X_test)
    else:
        num_evaluated_samples = args.num

X_train = np.concatenate((X_train[:, 0:1], np.log1p(X_train[:, 1:])), axis=1)
X_test = np.concatenate((X_test[:, 0:1], np.log1p(X_test[:, 1:])), axis=1)
scaler = joblib.load(os.path.join(asset_model_dir, "handcraft_scaler.save"))

X_train = scaler.transform(X_train[:, 1:])
X_test = scaler.transform(X_test[:, 1:])

input_dim = X_train.shape[1]
num_layers = 4
num_units = 200
output_dim = 11


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(np.array(X_train).astype(np.float32))
        self.y = torch.from_numpy(np.array(y_train).astype(int))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


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


train = Data(X_train, Y_train)
trainloader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2)

testdata = Data(X_test, Y_test)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

mlp_classifier = Network()
mlp_classifier.load_state_dict(torch.load(os.path.join(asset_model_dir, "mlp_handcraft.pt")))
mlp_classifier.eval()
prediction = []
ground_truth = []
for i, testdata in enumerate(testloader):
    inputs, labels = testdata
    pred = mlp_classifier(inputs)
    prediction.append(pred)
    ground_truth.append(labels)
prediction_results = torch.cat(prediction).detach().numpy()
ground_truth_results = torch.cat(ground_truth).detach().numpy()
print(prediction_results.shape)
print(ground_truth_results.shape)
accuracy = accuracy_score(
    np.argmax(prediction_results, axis=1),
    ground_truth_results,
)
print(f"Original testing accuracy: {accuracy}")

print("Start attacking")

prediction = []
ground_truth = []
smt_cross_entropy_results = []
pgd_cross_entropy_results = []
orig_cross_entropy_results = []
adv_attempts = 0
success_attempts = 0
total = 0
correct = 0

pkt_num_change = 0
wrong_sample_idx = 0
correct_sample_idx = 0

CE_loss = torch.nn.CrossEntropyLoss()
# Sequential
manager = mp.Manager()
smt_return_dict = manager.dict()
if attack_train:
    loader = trainloader
else:
    loader = testloader

slice_data_volumn = np.ceil(X_train.shape[0] / 3)
time_start = time.time()
total_num_adv_samples = 0
total_generated_outputs = 0
for i, testdata in enumerate(loader):
    if attack_train and slice_id == 0:
        if i >= slice_data_volumn:
            break

    if attack_train and slice_id == 1:
        if i < slice_data_volumn:
            continue
        if i >= 2 * slice_data_volumn:
            break

    if attack_train and slice_id == 2:
        if i < 2 * slice_data_volumn:
            continue

    inputs, labels = testdata

    pred = mlp_classifier(inputs)

    if np.argmax(pred.detach().numpy()) == labels.detach().numpy()[0]:
        (
            adv_loss,
            orig_loss,
            attack_success,
            adv_flow_list,
            considered_important_feature_superlist,
        ) = pgd_attack_RANDOM_INIT_v3(
            model=mlp_classifier,
            index=int(X_test_orig[i][0]),
            features=inputs,
            labels=labels,
            only_src=only_src,
        )
        # print(adv_loss, orig_loss)
        # print(orig_loss)
        orig_cross_entropy_results.append(orig_loss)
        smt_cross_entropy_results.append(adv_loss)
        total_generated_outputs += len(adv_flow_list)
        if attack_success:
            success_attempts += 1
            total_num_adv_samples += len(adv_flow_list)
            if attack_train:
                for adv_flow_idx, final_adv_flow in enumerate(adv_flow_list):
                    # TODO!!
                    final_adv_flow.to_csv(
                        os.path.join(
                            result_folder,
                            "adv_samples",
                            f"adv_{int(X_test_orig[i][0])}_{adv_flow_idx}.csv",
                        ),
                        index=False,
                    )
            with open(
                os.path.join(result_folder, "important_features.json"), "a"
            ) as fp:
                json.dump(considered_important_feature_superlist, fp)
                fp.write("\n")
        else:
            if attack_train:
                for adv_flow_idx, final_adv_flow in enumerate(adv_flow_list):
                    # TODO!!
                    final_adv_flow.to_csv(
                        os.path.join(
                            result_folder,
                            "unsuccess_samples",
                            f"{int(X_test_orig[i][0])}_{adv_flow_idx}.csv",
                        ),
                        index=False,
                    )
        adv_attempts += 1

        print(f"Progress: {adv_attempts} / {num_evaluated_samples}")
        if adv_attempts % 10 == 0:
            now = time.time()
            time_elapsed = now - time_start
            if total_num_adv_samples != 0:
                l3 = f"samples attacked: {adv_attempts}, ASR: {success_attempts / adv_attempts}, speed: {total_num_adv_samples/time_elapsed}\n"
            else:
                l3 = f"samples attacked: {adv_attempts}, ASR: {success_attempts / adv_attempts}, speed: {total_num_adv_samples/time_elapsed}\n"
            print(l3)
            asr_path = os.path.join(result_folder, "result.txt")
            file = open(asr_path, "a")
            file.write(l3)
            file.close()

        if args.attack == "test" and adv_attempts == num_evaluated_samples:
            break

orig_cross_entropy_results = np.array(orig_cross_entropy_results)
smt_cross_entropy_results = np.array(smt_cross_entropy_results)

np.savez(
    os.path.join(result_folder, "loss.npz"),
    orig=orig_cross_entropy_results,
    smt=smt_cross_entropy_results,
    pgd=pgd_cross_entropy_results,
)

now = time.time()
time_elapsed = now - time_start
l3 = f"Summary: ASR: {success_attempts / adv_attempts}, speed: {total_num_adv_samples/time_elapsed}\n"
print(l3)
asr_path = os.path.join(result_folder, "result.txt")
file = open(asr_path, "a")
file.write(l3)
file.close()
