import json
import glob
import matplotlib.pyplot as plt
import numpy as np

'''
TODO:
    - Implement figure saving in the proper place, maybe with
      a figpath variable.
    - Implement a better visualization.
'''
def get_name_from_path(filepath):
    # This should be done using OS instead of replacing like a dumbdumb.
    return filepath.split("/")[-1].split(".")[0]

def get_plot_params(filepaths):
    '''
    TODO: write docstring.
    '''
    vmin, vmax = np.Inf, np.NINF
    for filepath in filepaths:
        with open(filepath) as fp:
            generation = json.load(fp)
        
        for doc in generation.values():
            if doc["performance"] is None:
                continue

            if doc["performance"] > vmax:
                vmax = doc["performance"]
            if doc["performance"] < vmin:
                vmin = doc["performance"]

    return vmin, vmax

def plot_generation(filepath, xlims=None, ylims=None, vmin=None, vmax=None):
    with open(filepath) as fp:
        generation = json.load(fp)
    
    points = np.zeros((len(generation), 2))
    colors = np.zeros(len(generation))
    # How to deal with None's?
    i = 0
    for doc in generation.values():
        if doc["performance"] is None:
            np.delete(points, i, axis=0)
            np.delete(colors, i)
            # points[i, :] = doc["centroid"]
            # colors[i] = 0
            # i += 1
        else:
            points[i, :] = doc["centroid"]
            colors[i] = doc["performance"]
            i += 1
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, vmin=vmin, vmax=vmax)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.colorbar(scatter)

    title = get_name_from_path(filepath).replace("_", " ")
    ax.set_title(title)
    plt.savefig(filepath.replace(".json", ".jpg"), format="jpg")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    filepaths = list(glob.glob('./generation_*.json'))

    print("Getting the plot parameters")
    vmin, vmax = get_plot_params(filepaths)

    for i, filepath in enumerate(filepaths):
        print(f"{i+1}/{len(filepaths)}")
        plot_generation(
            filepath, xlims=(- 2*np.pi, 2*np.pi), ylims=(-2*np.pi, 2*np.pi),
            vmin=vmin, vmax=vmax
        )
