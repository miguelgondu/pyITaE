import numpy as np
from pymelites.visualizing_generations import plot_generations

plot_generations("./update_updating_*.json", partition={
    "feature_a": (-2*np.pi, 2*np.pi, 100),
    "feature_b": (-np.pi, 3*np.pi, 100)
})
