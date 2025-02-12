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

def plot_robustfication(mlp, rf, tf, cnn, out_path):
    
    plt.figure(figsize=(8, 3.5))
    
    x = np.arange(len(mlp))
    y = np.arange(len(tf))

    plt.figure(figsize=(6, 5))
    plt.plot(x, mlp, "--+", linewidth=3, markersize=12, label = "MLP")
    plt.plot(x, rf, "--v", linewidth=3, markersize=12, label = "RF")
    plt.plot(y, tf, "--o", linewidth=3, markersize=12, label = "TF")
    plt.plot(x, cnn, "--*", linewidth=3, markersize=12, label = "CNN")

    plt.xticks(y, ["Vanilla", "S1", "S2", "S3", "S4", "S5"]) 
    plt.ylabel("ASR (%)")
    legend = plt.legend()
    legend.get_frame().set_facecolor('grey')
    legend.get_frame().set_alpha(0.1)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    
for threat_short in ["end"]:
    mlp = []
    rf = []
    tf = []
    cnn = []
    for i in range(4):
        file_path = os.path.join(result_dir, f"mlp_{threat_short}_pants_r{i}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        mlp.append(ASR)
    for i in range(4):
        file_path = os.path.join(result_dir, f"rf_{threat_short}_pants_r{i}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        rf.append(ASR)
    for i in range(6):
        file_path = os.path.join(result_dir, f"tf_{threat_short}_pants_r{i}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        tf.append(ASR)
    for i in range(4):
        file_path = os.path.join(result_dir, f"cnn_{threat_short}_pants_r{i}", "result.txt")
        
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            
        ASR = float(last_line.split(",")[0].split(" ")[-1])
        cnn.append(ASR)
        
    if threat_short == "end":
        threat_model = "end-host"
    elif threat_short == "in":
        threat_model = "in-host"
    else:
        raise ValueError("Invalid threat model")

    out_path = os.path.join(args.figure_dir, f"robustfication_{threat_model}.pdf")
    plot_robustfication(np.array(mlp), np.array(rf), np.array(tf), np.array(cnn), out_path)
    