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
    features = x[:2]
    performance = 0
    return performance, features

kernel = 1 * RBF(length_scale=1) + WhiteKernel(noise_level=np.log(2))

itae = ITAE(
    path="new_prior.json",
    deploy=deploy,
    kernel=kernel,
    # goal=15,
    # distance_to_goal=5,
    comment="updating_plane",
    max_iterations=10
)

# itae = ITAE(
#     path="rastrigin_archive.json",
#     deploy=deploy
# )

try:
    itae.run()
except KeyboardInterrupt:
    pass