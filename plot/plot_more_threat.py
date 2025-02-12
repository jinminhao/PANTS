import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
import os
from barplot_vertical import draw_barplot

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
parser.add_argument("--figure_dir", type=str, help="figure directory")
args = parser.parse_args()
result_dir = args.folder

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size': 22})

def plot_more_threat(metric_data, out_path):
    candidate_hatchs = ['+++', 'OOO', 'xxx', '***', '...', '///', 'ooo']
    candidate_color = ['#B3BC95', '#D5E4A6', '#ADBED6', '#3E5D70', "#E57259", "#FAAF78", ]
    draw_barplot(metric_data, candidate_color, candidate_hatchs, os.path.join(out_path, "threat.pdf"), 'ASR (%)')


results = {
    "mlp": [
        ["mlp_end_vanilla_d40_a40_i40_s0", "mlp_end_pants_d40_a40_i40_s0"],
        ["mlp_end_vanilla_d39_a6_i8_s0", "mlp_end_pants_d39_a6_i8_s0"],
        ["mlp_end_vanilla_d28_a19_i36_s0", "mlp_end_pants_d28_a19_i36_s0"],
        ["mlp_end_vanilla_d34_a38_i19_s0", "mlp_end_pants_d34_a38_i19_s0"],
        ["mlp_end_vanilla_d28_a25_i0_s0", "mlp_end_pants_d28_a25_i0_s0"],
        ["mlp_end_vanilla_d40_a40_i40_s40", "mlp_end_pants_d40_a40_i40_s40"],
        ["mlp_end_vanilla_d21_a5_i25_s15", "mlp_end_pants_d21_a5_i25_s15"],
        ["mlp_end_vanilla_d16_a40_i19_s20", "mlp_end_pants_d16_a40_i19_s20"],
    ],
    "cnn": [
        ["cnn_end_vanilla_d40_a40_i40_s0", "cnn_end_pants_d40_a40_i40_s0"],
        ["cnn_end_vanilla_d39_a6_i8_s0", "cnn_end_pants_d39_a6_i8_s0"],
        ["cnn_end_vanilla_d28_a19_i36_s0", "cnn_end_pants_d28_a19_i36_s0"],
        ["cnn_end_vanilla_d34_a38_i19_s0", "cnn_end_pants_d34_a38_i19_s0"],
        ["cnn_end_vanilla_d28_a25_i0_s0", "cnn_end_pants_d28_a25_i0_s0"],
        ["cnn_end_vanilla_d40_a40_i40_s40", "cnn_end_pants_d40_a40_i40_s40"],
        ["cnn_end_vanilla_d21_a5_i25_s15", "cnn_end_pants_d21_a5_i25_s15"],
        ["cnn_end_vanilla_d16_a40_i19_s20", "cnn_end_pants_d16_a40_i19_s20"],
    ]
}

metric_data = []
for model in ["mlp", "cnn"]:
    result = {
        "before": {},
        "after": {}
    }
    for i in range(8):
        x = results[model][i][0].split("_")[-4:]
        label = f"({x[0][1:]},{x[1][1:]},{x[2][1:]},{x[3][1:]})"
        for j in range(2):
            file_path = os.path.join(result_dir, results[model][i][j], "result.txt")
            with open(file_path, 'r') as f:
                last_line = f.readlines()[-1]
            
            ASR = np.array([float(last_line.split(",")[0].split(" ")[-1]) * 100])
            if j == 0:
                result["before"][label] = ASR
            else:
                result["after"][label] = ASR
                
    metric_data.append(result)
    

plot_more_threat(metric_data, args.figure_dir)
