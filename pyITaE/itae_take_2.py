import json
import numpy as np
import GPy

def load_map(path):
    """
    For now, it is implemented with hashmaps (dicts), it could be implemented with heaps
    to make finding the max a logarithmic operation instead of a linear one.
    """
    with open(path) as fp:
        docs = json.load(fp)

    perf_map, behaviors_map, controllers = {}, {}, {}
    for doc in docs.values():
        centroid = doc["centroid"]
        perf_map[centroid] = doc["performance"]
        behaviors_map[centroid] = doc["features"]
        controllers[centroid] = doc["solution"]

    return perf_map, behaviors_map, controllers

def deploy(controller):
    """
    This is the core. Here, the agent should be presented with a game, the rollouts should be
    recorded, and from there the optimization should take place.

    This one should be way more general, but for now I'll handcode the deploy procedure.
    """
    return 0, 0

def check_stopping_condition(performance, recorded_perfs):
    """
    Check the paper and their original implementation.
    """
    return False

def get_next_centroid(real_map):
    current_centroid, current_max = None, -np.Inf
    for centroid, performance in real_map.items():
        if performance is not None and performance > current_max:
            current_centroid = centroid

    return current_centroid

def update_real_map(model, perf_map, behaviors_map):
    centroids = []
    behaviors = []
    map_performances = []
    means, variances = [], []
    for centroid, behavior in behaviors_map.items():
        centroids.append(centroid)
        behaviors.append(behavior)
        map_performances.append(perf_map[centroid])

        mean, variance = model.predict(np.array([behavior]))[0]
        means.append(mean)
        variances.append(variance)
    
    real_values = np.array(means) + np.array(map_performances)
    real_map = {
        centroid: real_values[i] for i, centroid in enumerate(centroids)
    }
    return real_map


def itae(path):
    """
    The algorithm itself.
    """
    perf_map, behaviors_map, controllers = load_map(path)

    real_map = perf_map.copy()

    # I could change this to a "get next index" function.
    # next_index = np.argmax(real_map)
    next_centroid = get_next_centroid(real_map)

    next_controller = controllers[next_centroid]
    X = []
    Y = []
    recorded_perfs = []

    while True:
        performance, behavior = deploy(next_controller)
        recorded_perfs.append(performance)

        stopping_condition = check_stopping_condition(performance, recorded_perfs)
        if stopping_condition:
            break

        dimension = len(behavior)
        kernel = GPy.kern.Matern52(dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension,np.sqrt(0.1))

        X.append(behavior)
        Y.append(performance - perf_map[next_index])

        m = GPy.models.GPRegression(X, Y, kernel)

        # If this is the only use of behaviors_map, then it should just be a np.array.
        mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        real_map = update_real_map(m, perf_map, behaviors_map)

        next_centroid = get_next_centroid(real_map)
        next_controller = controllers[next_centroid]
