import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
import os

import argparse

def get_transferability(path):
    with open(path, "r") as file:
        content = file.read()
    return round(float(content.split(" ")[-1]) * 100, 2)

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-f", "--folder", type=str, help="result folder name")
parser.add_argument("--figure_dir", type=str, help="figure directory")
args = parser.parse_args()
result_dir = args.folder

mlp_to_cnn = get_transferability(os.path.join(args.folder, "mlp_to_cnn.txt"))
mlp_to_rf = get_transferability(os.path.join(args.folder, "mlp_to_rf.txt"))
mlp_to_tf = get_transferability(os.path.join(args.folder, "mlp_to_tf.txt"))
rf_to_cnn = get_transferability(os.path.join(args.folder, "rf_to_cnn.txt"))
rf_to_mlp = get_transferability(os.path.join(args.folder, "rf_to_mlp.txt"))
rf_to_tf = get_transferability(os.path.join(args.folder, "rf_to_tf.txt"))
tf_to_cnn = get_transferability(os.path.join(args.folder, "tf_to_cnn.txt"))
tf_to_mlp = get_transferability(os.path.join(args.folder, "tf_to_mlp.txt"))
tf_to_rf = get_transferability(os.path.join(args.folder, "tf_to_rf.txt"))
cnn_to_mlp = get_transferability(os.path.join(args.folder, "cnn_to_mlp.txt"))
cnn_to_rf = get_transferability(os.path.join(args.folder, "cnn_to_rf.txt"))
cnn_to_tf = get_transferability(os.path.join(args.folder, "cnn_to_tf.txt"))

mpl.rcParams.update({'font.size': 22})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

model = ["MLP", "RF", "TF", "CNN"]
reversed_model = model[::-1]

harvest = np.array([[mlp_to_cnn, mlp_to_tf, mlp_to_rf, 81.12],[rf_to_cnn, rf_to_tf, 66.50, rf_to_mlp], [tf_to_cnn, 99.80, tf_to_rf, tf_to_mlp], [94.25, cnn_to_tf, cnn_to_rf, cnn_to_mlp]])

plt.figure(figsize=(10,7))
im = plt.imshow(harvest, cmap="RdYlGn", vmin=0, vmax=100)
cbar = plt.colorbar(im)
cbar.set_label("Transferability (%)")

# Show all ticks and label them with the respective list entries
plt.xticks(np.arange(len(model)), labels=reversed_model, rotation=0)
plt.yticks(np.arange(len(model)), labels=model)

plt.xlabel("Adversarial for")
plt.ylabel("Generated against")
# Loop over data dimensions and create text annotations.
for i in range(len(model)):
    for j in range(len(reversed_model)):
        text = plt.text(j, i, f"{harvest[i, j]}%",
                       ha="center", va="center", color="black")

plt.savefig(os.path.join(args.figure_dir, "transferability.pdf"), dpi=300, bbox_inches = 'tight')
plt.show()
plt.close()