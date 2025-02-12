import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
import os

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
parser.add_argument("--figure_dir", type=str, help="figure directory")
args = parser.parse_args()
result_dir = args.folder

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size': 22})

def plot_adv_train(result, out_path):
    color = {
        "mlp": "r",
        "rf": "c",
        "tf": "b",
        "cnn": "g"
    }
    marker = {
        "adv_train": "*",
        "pgd_train": "v",
        "pants_robust": "X",
        "amoeba_robust": "s"
    }
    for model in ["mlp", "rf", "tf", "cnn"]:
        for robustification in ["adv_train", "pgd_train", "pants_robust", "amoeba_robust"]:
            if model == "rf" and robustification == "adv_train":
                continue
            
            if robustification == "adv_train":
                title = "Adversarilly-trained"
            elif robustification == "pgd_train":
                title = "PGD-robustified"
            elif robustification == "pants_robust":
                title = "PANTS-robustified"
            elif robustification == "amoeba_robust":
                title = "Amoeba-robustified"
            plt.plot(result[model][robustification]["Accuracy"] * 100, result[model][robustification]["ASR"] * 100, 
                     color=color[model], marker=marker[robustification], markersize=15, alpha=0.9, label=f"{title} {model}")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("ASR (%)")
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

result = {}
for model in ["mlp", "rf", "tf", "cnn"]:
    result[model] = {}
    for robustification in ["adv_train", "pgd_train", "pants_robust", "amoeba_robust"]:
        if model == "rf" and robustification == "adv_train":
            continue
        file_path = os.path.join(result_dir, f"{robustification}_{model}", "result.txt")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            first_line = lines[0]
            last_line = lines[-1]
            
        Accuracy = float(first_line.split(" ")[-1])
        ASR = float(last_line.split(",")[0].split(" ")[-1])

        result[model][robustification] = {
            "Accuracy": Accuracy,
            "ASR": ASR
        }

plot_adv_train(result, os.path.join(args.figure_dir, "adv_train.pdf"))