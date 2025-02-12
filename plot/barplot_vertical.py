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

def draw_barplot(template_datas,
        candidate_color,
        candidate_hatchs,
        saved_path,
        xlabel,
        figsize=(5, 9)
        # figsize=(9, 4)
        ):
    # refactor the template_data based on metric
    modified_data = {}
    modified_data2 = {}
    colors = {}
    plt.figure(figsize=figsize)

    template_data = template_datas[0]
    template_data2 = template_datas[1]
    assert(len(candidate_color) >= len(template_data.keys()))

    total_boxes = 0
    for index, key_method in enumerate(template_data.keys()):
        for key_metric in template_data[key_method]:
            if key_metric not in modified_data:
                modified_data[key_metric] = {}
                modified_data2[key_metric] = {}
            modified_data[key_metric][key_method] = template_data[key_method][key_metric]
            modified_data2[key_metric][key_method] = template_data2[key_method][key_metric]
            total_boxes += 1
        colors[key_method] = candidate_color[index]

    num_metric = len(modified_data.keys())
    num_method = len(template_data.keys())

    ###########
    mpl.rcParams.update({'font.size': 16})
    
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    modified_datas = [modified_data, modified_data2]
    
    for i, model in enumerate(["MLP", "CNN"]):
    
        width = 0.35
        gap = 0.8
        pos = 0.0
        yticks = []
        ytick_labels = []

        for metric in modified_datas[i].keys():
            poses = []
            j = 0
            handles = []
            labels = []
            for method in modified_datas[i][metric].keys():
                data = modified_datas[i][metric][method]
                handles.append(axs[i].barh(
                    pos,
                    np.mean(data),
                    width,
                    
                    xerr=np.std(data) / math.sqrt(len(data)),
                    # xerr=0,
                    color="None",
                    hatch=candidate_hatchs[j],
                    edgecolor=candidate_color[j]
                    ))
                labels.append(method)
                poses.append(pos)
                pos -= width
                j += 1
            yticks.append((poses[0] + poses[-1]) / 2)
            ytick_labels.append(metric)

            pos -= gap
            
        axs[i].set_title(model)

    
    legend = plt.legend(handles, labels, fontsize=18, labelspacing=0.2,)
    # Set the transparency for the legend
    legend.get_frame().set_facecolor('grey')
    legend.get_frame().set_alpha(0.1)  # Set the transparency (0: transparent, 1: opaque)
    plt.legend(handles, labels, fontsize=18, labelspacing=0.2, bbox_to_anchor=(-2.3, 1.1), loc='upper left')

    fig.supxlabel(xlabel, fontsize=22, y=0.03)
    plt.tight_layout()

    plt.yticks(yticks, ytick_labels, fontsize=22)

    
    plt.savefig(saved_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

