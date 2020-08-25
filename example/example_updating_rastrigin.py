import numpy as np
from pyITaE.itae import ITAE

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
    n = len(x)
    A = 3
    performance = - 10 -(A*n + np.sum(x ** 2 - A*np.cos(2*np.pi*x)))
    return performance, features

# itae = ITAE(
#     path="rastrigin_archive.json",
#     deploy=deploy,
#     goal=-100,
#     distance_to_goal=5)

itae = ITAE(
    path="rastrigin_archive.json",
    deploy=deploy
)

itae.run()