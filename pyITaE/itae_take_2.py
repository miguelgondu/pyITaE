import json
import numpy as np
import GPy
from collections import defaultdict

def to_json_writable(dict_):
    """
    TODO: embetter this function
    """
    new_dict = {
        str(k): v for k, v in dict_.items()
    }
    return new_dict

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
        if doc["performance"] is not None:
            perf_map[centroid] = doc["performance"]
            behaviors_map[centroid] = doc["features"]
            controllers[centroid] = doc["solution"]

    return perf_map, behaviors_map, controllers

def check_stopping_condition(real_map, recorded_perfs, alpha):
    """
    Returns whether the recorded performances we have seen
    are bigger than a bound (alpha times the best new estimate
    of the real performance).
    """
    bound = alpha*max(real_map.values())
    max_perf = max(recorded_perfs)
    print(f"max recorded performance: {max_perf}, bound: {bound}")
    return max(recorded_perfs) > bound

def get_next_centroid(real_map):
    current_centroid, current_max = None, -np.Inf
    for centroid, performance in real_map.items():
        if performance > current_max:
            current_centroid = centroid
            current_max = performance
            print(f"current centroid: {current_centroid}, performance: {performance}")

    print(f"Next centroid: {current_centroid}.")
    return current_centroid

def update_real_map(model, perf_map, behaviors_map):
    centroids = []
    behaviors = []
    map_performances = []
    for centroid, behavior in behaviors_map.items():
        centroids.append(centroid)
        behaviors.append(behavior)
        performance = perf_map[centroid]
        assert performance is not None
        map_performances.append(perf_map[centroid])

    # print(f"behaviors: {behaviors}")
    # _ = input("Press enter to continue. Debugging behaviors.")
    behaviors = np.array(behaviors)
    # print(f"behaviors: {behaviors}, shape: {behaviors.shape}")
    # _ = input("Press enter to continue. Debugging behaviors.")

    mean, variance = model.predict(behaviors)
    # print(f"mean: {mean}")
    # _ = input("Press enter to continue. Debugging mean.")

    real_values = mean.T[0] + np.array(map_performances)
    # print(f"real values: {real_values}")
    # _ = input("Press enter to continue. Debugging real_values.")
    real_map = {
        centroid: real_values[i] for i, centroid in enumerate(centroids)
    }
    # print(f"real map: {real_map}")
    # _ = input("Press enter to continue. Debugging real_map.")
    return real_map


def itae(path, deploy, max_iterations=100):
    """
    The algorithm itself. Takes the path to the map outputted by pymelites
    and the deploy function, which should take a controller and return
    performance and behavior.

    TODO: test and visualize this with the rastrigin.
    """

    # Preamble: map loading and defining constants
    perf_map, behaviors_map, controllers = load_map(path)
    real_map = perf_map.copy()

    tested_centroids = defaultdict(int)

    X = []
    Y = []
    recorded_perfs = []

    alpha = 0.8

    best_controller, best_performance = None, -np.Inf

    # The main loop
    updates = 0
    while True:
        # Get the next controller to test, check if it
        # has been tested in the past.
        next_centroid = get_next_centroid(real_map)
        next_controller = controllers[next_centroid]
        if next_centroid in tested_centroids:
            # What do?
            # I could retry it with some probability.
            pass

        print(f"Deploying the controller {next_controller}")
        performance, behavior = deploy(next_controller)
        recorded_perfs.append(performance)
        if best_performance < performance:
            best_performance = performance
            best_controller = next_controller
            print(f"New best (real) performance found: {performance}")
            print(f"Associated controller: {best_controller}")
        tested_controllers[next_centroid] += 1
        print(f"Performance of that controller: {performance}")

        dimension = len(behavior)
        print(dimension)
        print(f"dimension")
        _ = input("Press enter to continue.")

        kernel = GPy.kern.Matern52(input_dim=dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension, np.sqrt(0.1))

        X.append(behavior)
        Y.append(performance - perf_map[next_centroid])

        print(f"X: {X}, Y: {Y}")
        m = GPy.models.GPRegression(
            np.array(X),
            np.array([Y]).T,
            kernel
        )

        # If this is the only use of behaviors_map, then it should just be a np.array.
        # mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        real_map = update_real_map(m, perf_map, behaviors_map)
        print(f"New real map: {real_map}.")
        update += 1

        # Saving the update for visualization
        # TODO: add generality
        with open(f"./update_{update}.json", "w") as fp:
            json.dump(
                to_json_writable(real_map),
                fp
            )

        # Check stopping conditions
        stopping_condition = check_stopping_condition(real_map, recorded_perfs, alpha)
        if stopping_condition:
            print("The stopping condition has been achieved. Stopping.")
            break
        if updates > max_iterations:
            break
    
    # TODO: return the new best performing controller.

