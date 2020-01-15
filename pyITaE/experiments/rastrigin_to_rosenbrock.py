"""
In this script we test the ITAE algorithm by first simulating
on the rastrigin 6D function, and then test by "deploying" using
the rosenbrock 6D function.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import glob

from pyITaE.itae_take_2 import itae
from pyITaE.visualize_updates import plot_updates
from pymelites.map_elites import MAP_Elites

# Computing the archive using the rastrigin function.
DIMENSIONS = 6
A = 10

def random_solution():
    return np.random.uniform(-2*np.pi, 2*np.pi, DIMENSIONS)

def random_selection(X):
    return random.choice(X)

def random_variation(x, scale=1):
    return x + np.random.normal(0, scale, size=x.size)

def performance(x):
    # The rastrigin function, in this case.
    # Since MAP_elites solves maximizations, I consider
    # the "-rastrigin" function. Thus, the max is
    # actually 0.

    n = len(x)
    return -(A*n + np.sum(x ** 2 - A*np.cos(2*np.pi*x)))

def feature_descriptor(x):
    return x[:2]

def simulate(x):
    p = performance(x)
    features = x[:2]
    return features, p


partitions = {
    "x1": (-2*np.pi, 2*np.pi, 100),
    "x2": (-2*np.pi, 2*np.pi, 100)
}

map_elites = MAP_Elites(
    random_solution=random_solution,
    random_selection=random_selection,
    random_variation=random_variation,
    simulate=simulate
)

map_elites.create_cells(
    partition=[(-2*np.pi, 2*np.pi, 75), (-2*np.pi, 2*np.pi, 75)],
    amount_of_elites=3
)

# map_elites.compute_archive(2500, 100, generation_path='.', save_each_gen=False)

# Implementing the deployment with the rosenbrock function

def deploy(x):
    performance = 0
    for i in range(len(x) - 1):
        performance += 10*((x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
    
    return -performance, x[:2]

try:
    itae("./generation_02499.json", deploy)
except KeyboardInterrupt:
    pass

limits = (-2*np.pi, 2*np.pi)
plot_updates(glob.glob("./update_*.json"), limits, limits)
