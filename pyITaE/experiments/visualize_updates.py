import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from visualize_generation import get_name_from_path

def get_plot_params(doc):
    min_, max_ = np.Inf, -np.Inf
    for performance in doc.values():
        if performance < min_:
            min_ = performance
        
        if performance > max_:
            max_ = performance
    
    return min_, max_

def process_file(filepath):
    with open(filepath) as fp:
        doc = json.load(fp)
    
    vmin, vmax = get_plot_params(doc)

    points = np.zeros((len(doc), 2))
    colors = np.zeros(len(doc))

    i = 0
    for k, v in doc.items():
        points[i, :] = json.loads(k.replace("(", "[").replace(")", "]"))
        colors[i] = v
        i += 1
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, vmin=vmin, vmax=vmax)
    ax.set_xlim((-2*np.pi, 2*np.pi))
    ax.set_ylim((-2*np.pi, 2*np.pi))
    plt.colorbar(scatter)

    title = get_name_from_path(filepath).replace("_", " ")
    ax.set_title(title)
    plt.savefig(filepath.replace(".json", ".jpg"), format="jpg")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    filepaths = glob.glob("./update_*.json")

    for filepath in filepaths:
        process_file(filepath)