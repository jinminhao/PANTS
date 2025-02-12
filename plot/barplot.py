import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math, json, copy
from pyparsing import Or

from sklearn import metrics
from collections import OrderedDict
import statistics

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.use("Agg")

def draw_barplot(template_data,
        candidate_color,
        candidate_hatchs,
        saved_path,
        ylabel,
        figsize=(6, 3.5)):
    # refactor the template_data based on metric
    modified_data = {}
    colors = {}
    plt.figure(figsize=figsize)
    # plt.figure(figsize=(8, 3.5))
    # plt.figure(figsize=(12,10))


    assert(len(candidate_color) >= len(template_data.keys()))

    total_boxes = 0
    for index, key_method in enumerate(template_data.keys()):
        for key_metric in template_data[key_method]:
            if key_metric not in modified_data:
                modified_data[key_metric] = {}
            modified_data[key_metric][key_method] = template_data[key_method][key_metric]
            total_boxes += 1
        colors[key_method] = candidate_color[index]

    num_metric = len(modified_data.keys())
    num_method = len(template_data.keys())

    ###########
    mpl.rcParams.update({'font.size': 22})
    
    width = 0.35
    gap = 0.8
    pos = 0.0
    xticks = []
    xtick_labels = []

    for metric in modified_data.keys():
        poses = []
        j = 0
        handles = []
        labels = []
        for method in modified_data[metric].keys():
            data = modified_data[metric][method]
            handles.append(plt.bar(
                pos,
                np.mean(data),
                width,
                yerr=np.std(data) / math.sqrt(len(data)),
                color="None",
                hatch=candidate_hatchs[j],
                edgecolor=candidate_color[j]
                ))
            labels.append(method)
            poses.append(pos)
            pos += width
            j += 1
        xticks.append((poses[0] + poses[-1]) / 2)
        xtick_labels.append(metric)

        pos += gap

    plt.tight_layout()
    # legend = plt.legend(handles, labels, fontsize=18, labelspacing=0.2)
    # # Set the transparency for the legend
    # legend.get_frame().set_facecolor('grey')
    # legend.get_frame().set_alpha(0.1)  # Set the transparency (0: transparent, 1: opaque)
    plt.legend(handles, labels, fontsize=18, labelspacing=0.2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel(ylabel)
    # plt.ylabel("JS divergence")
    # plt.ylabel("Accuracy")
    plt.xticks(xticks, xtick_labels)
    # plt.ylim(0, 100) 
    # plt.tick_params(axis="x", rotation=15)
    plt.savefig(saved_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()