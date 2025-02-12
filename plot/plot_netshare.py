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

def plot_netshare(ASR_list, xticks, out_path):
    ASR_list = np.array(ASR_list)
    
    plt.figure(figsize=(8, 3.5))
    

    plt.figure(figsize=(6, 5))
    x = [f"S{i+1}" for i in range(len(ASR_list))]
    plt.plot(x, ASR_list * 100, 'ro-',  markersize=10, alpha=0.5, label="NetShare")

    plt.xticks(x, xticks)
    plt.ylim(ymin=0)
    plt.xlabel("Number of samples")
    plt.ylabel("ASR (%)")
    plt.legend()

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


xticks_dict = {
    "mlp": ["X44", "X59", "X66"],
    "rf": ["X44", "X59", "X66"],
    "tf": ["X44", "X81", "X116", "X147", "X168"],
    "cnn": ["X40", "X60", "X73"]
}

for model in ["mlp", "rf", "tf", "cnn"]:
    if model == "tf":
        round_num = 5
    else:
        round_num = 3
    
    ASR_list = []
    for i in range(round_num):
        
        file_path = os.path.join(result_dir, f"{model}_end_netshare_r{i+1}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        ASR_list.append(ASR)
    

    out_path = os.path.join(args.figure_dir, f"netshare_{model}.pdf")
    plot_netshare(ASR_list, xticks_dict[model], out_path)
    