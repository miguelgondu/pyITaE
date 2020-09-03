import numpy as np
from pyITaE.itae import ITAE
from pymelites.visualizing_generations import plot_generations

from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def deploy(x):
    """
    Every ITAE object takes a `deploy` function
    that evaluates elites `x` and returns
    either (performance, features) or (performance,
    features, metadata).

    In the case of Rastrigin, we're assuming the
    genotypes are 6D vectors, the features are
    the first two dimensions and the performance
    is a mellowed out version of Rastrigin.
    """
    x = np.array(x)
    x = x - 1 # Shifting points for even more of a dramatic change
    features = x[:2]
    n = len(x)
    A = 3

    # Rosenbrock
    performance = 0
    for i in range(len(x) - 1):
        performance += ((x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
    # performance = - 10 -(A*n + np.sum(x ** 2 - A*np.cos(2*np.pi*x)))
    # performance = -200
    return -performance, features

kernel = 1 * RBF(length_scale=1) + WhiteKernel(noise_level=np.log(2))

itae = ITAE(
    path="rastrigin_archive.json",
    deploy=deploy,
    kernel=kernel,
    goal=-100,
    distance_to_goal=5,
    comment="updating_rastrigin",
    max_iterations=100,
    retest=False
)

# itae = ITAE(
#     path="rastrigin_archive.json",
#     deploy=deploy
# )

try:
    itae.run()
except KeyboardInterrupt:
    pass