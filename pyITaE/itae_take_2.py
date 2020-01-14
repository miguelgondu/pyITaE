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
        centroid = tuple(doc["centroid"])
        perf_map[centroid] = doc["performance"]
        behaviors_map[centroid] = doc["features"]
        controllers[centroid] = doc["solution"]

    return perf_map, behaviors_map, controllers

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
    # centroids = []
    # behaviors = []
    # map_performances = []
    # means, variances = [], []
    # for centroid, behavior in behaviors_map.items():
    #     centroids.append(centroid)
    #     behaviors.append(behavior)
    #     map_performances.append(perf_map[centroid])

    #     mean, variance = model.predict(np.array([[behavior]]))[0]
    #     print(f"mean: {mean}")
    #     print(f"variance: {variance}")
    #     means.append(mean)
    #     variances.append(variance)
    
    # real_values = np.array(means) + np.array(map_performances)
    # real_map = {
    #     centroid: real_values[i] for i, centroid in enumerate(centroids)
    # }
    # return real_map

    centroids = []
    behaviors = []
    map_performances = []
    for centroid, behavior in behaviors_map.items():
        if behavior is not None:
            centroids.append(centroid)
            behaviors.append(behavior)
            performance = perf_map[centroid]
            assert performance is not None
            map_performances.append(perf_map[centroid])

    # print(f"behaviors: {behaviors}")
    # _ = input("Press enter to continue. Debugging behaviors.")
    behaviors = np.array(behaviors)
    print(f"behaviors: {behaviors}, shape: {behaviors.shape}")
    _ = input("Press enter to continue. Debugging behaviors.")

    mean, variance = model.predict(behaviors)
    print(f"mean: {mean}")
    _ = input("Press enter to continue. Debugging mean.")

    real_values = mean.T[0] + np.array(map_performances)
    print(f"real values: {real_values}")
    _ = input("Press enter to continue. Debugging real_values.")
    real_map = {
        centroid: real_values[i] for i, centroid in enumerate(centroids)
    }
    print(f"real map: {real_map}")
    _ = input("Press enter to continue. Debugging real_map.")
    return real_map


def itae(path, deploy):
    """
    The algorithm itself. Takes the path to the map outputted by pymelites
    and the deploy function, which should take a controller and return
    performance and behavior.
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
        _ = input("Press enter to continue.")
        print(f"Deploying the controller {next_controller}")
        performance, behavior = deploy(next_controller)
        print(f"Performance of that controller: {performance}")
        recorded_perfs.append(performance)

        stopping_condition = check_stopping_condition(performance, recorded_perfs)
        if stopping_condition:
            break

        dimension = len(behavior)
        print(dimension)
        print(f"dimension")
        _ = input("Press enter to continue.")

        kernel = GPy.kern.Matern52(input_dim=dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension, np.sqrt(0.1))

        X.append(behavior)
        Y.append(performance - perf_map[next_centroid])

        print(f"X: {X}")
        m = GPy.models.GPRegression(
            np.array(X),
            np.array([Y]),
            kernel
        )

        # If this is the only use of behaviors_map, then it should just be a np.array.
        # mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        real_map = update_real_map(m, perf_map, behaviors_map)
        print(f"New real map: {real_map}")

        next_centroid = get_next_centroid(real_map)
        next_controller = controllers[next_centroid]
