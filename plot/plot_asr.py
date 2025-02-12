import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
import os
from barplot import draw_barplot

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
parser.add_argument("--figure_dir", type=str, help="figure directory")
args = parser.parse_args()
result_dir = args.folder

for threat_short in ["end", "in"]:
    for application in ["app", "vpn", "vca"]:
        result_dict = {}
        result_dict["PANTS"] = {}
        for model in ["mlp", "rf", "tf", "cnn"]:
            file_path = os.path.join(result_dir, application, f"{model}_{threat_short}_vanilla", "result.txt")
            with open(file_path, 'r') as f:
                last_line = f.readlines()[-1]
            
            ASR = float(last_line.split(",")[0].split(" ")[-1]) * 100
            result_dict["PANTS"][model] = [ASR]

        if threat_short == "end":
            threat_model = "end_host"
        else:
            threat_model = "in_path"

        candidate_hatchs = ['+++', 'OOO', 'xxx', '***', '...', '///', 'ooo']
        candidate_color = ['#B3BC95',  '#ADBED6', '#3E5D70', '#D5E4A6',  "#E57259", "#FAAF78", ]
        draw_barplot(result_dict, candidate_color, candidate_hatchs, os.path.join(args.figure_dir, f"{application}_{threat_model}_vanilla.pdf"), 'ASR (%)')
