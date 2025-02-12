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

def plot_important_features(mlp, rf, out_path):
    
    plt.figure(figsize=(8, 3.5))
    
    x = [i * 2 for i in range(1, 12)]

    plt.plot(x, mlp * 100, 'o-', color="#B3BC95",  markersize=15, alpha=0.9, label='MLP')
    plt.plot(x, rf * 100, '*--', color="#3E5D70",  markersize=15, alpha=0.9, label='RF')

    plt.xlabel("Number of most important features (k)")
    plt.ylabel("ASR (%)")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    
for threat_short in ["end", "in"]:
    mlp = []
    rf = []
    for num_important_feature in [i * 2 for i in range(1, 12)]:
        file_path = os.path.join(result_dir, f"mlp_{threat_short}_{num_important_feature}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        mlp.append(ASR)
        
    for num_important_feature in [i * 2 for i in range(1, 12)]:
        file_path = os.path.join(result_dir, f"rf_{threat_short}_{num_important_feature}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        rf.append(ASR)
        
    if threat_short == "end":
        threat_model = "end-host"
    elif threat_short == "in":
        threat_model = "in-host"
    else:
        raise ValueError("Invalid threat model")

    out_path = os.path.join(args.figure_dir, f"important_features_{threat_model}.pdf")
    plot_important_features(np.array(mlp), np.array(rf), out_path)
    