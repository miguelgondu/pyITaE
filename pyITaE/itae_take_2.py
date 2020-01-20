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
    max_perf = max(recorded_perfs.values())
    print(f"max recorded performance: {max_perf}, bound: {bound}")
    return max_perf > bound

def get_first_centroid(real_map):
    current_centroid, current_max = None, -np.Inf
    for centroid, performance in real_map.items():
        if performance > current_max:
            current_centroid = centroid
            current_max = performance
            print(f"current centroid: {current_centroid}, performance: {performance}")

    print(f"Next centroid: {current_centroid}.")
    return current_centroid

def update_real_and_variance_maps(model, perf_map, behaviors_map):
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
    variance_values = variance.T[0]
    # print(f"real values: {real_values}")
    # _ = input("Press enter to continue. Debugging real_values.")

    # Updating the real and variance maps.
    real_map = {}
    variance_map = {}
    for i, centroid in enumerate(centroids):
        real_map[centroid] = real_values[i]
        variance_map[centroid] = variance_values[i]

    # print(f"real map: {real_map}")
    # _ = input("Press enter to continue. Debugging real_map.")
    return real_map, variance_map

def acquisition(real_map, variance_map, kappa):
    if variance_map == None:
        return get_first_centroid(real_map)

    next_centroid, best_bound = None, -np.Inf
    for centroid in real_map:
        bound = real_map[centroid] + kappa * variance_map[centroid]
        if bound > best_bound:
            next_centroid = centroid
            best_bound = bound
    print(f"Next centroid: {next_centroid}, value: {best_bound}")
    return next_centroid

def itae(path, deploy, max_iterations=100, retest=True, comment=""):
    """
    The algorithm itself. Takes the path to the map outputted by pymelites
    and the deploy function, which should take a controller and return
    performance and behavior.

    TODO: write a complete docstring. Refactor into a class.
    """

    # Preamble: map loading and defining constants
    perf_map, behaviors_map, controllers = load_map(path)
    real_map = perf_map.copy()

    tested_centroids = defaultdict(int)

    X = []
    Y = []
    recorded_perfs = {}
    recorded_behaviors = {}

    alpha = 0.85
    kappa = 0.03

    best_controller, best_performance = None, -np.Inf
    variance_map = None

    # The main loop
    updates = 0
    while True:
        print("-"*80)
        print(" " * 30 + f"Update {updates}" + " " * 30)
        to_append_to_X = None
        to_append_to_Y = None

        # Get the next controller to test, check if it
        # has been tested in the past.
        next_centroid = acquisition(real_map, variance_map, kappa)
        next_controller = controllers[next_centroid]

        if next_centroid in tested_centroids:
            if not retest:
                print("I have seen this controller before and I'm not retesting it. I'm assuming it will have the same real performance.")
                print(f"Using previous behavior: {behaviors_map[next_centroid]}")
                print(f"Using previous recorded performance: {recorded_perfs[next_centroid]}")
                to_append_to_X = recorded_behaviors[next_centroid]
                to_append_to_Y = recorded_perfs[next_centroid]
                dimension = len(to_append_to_X)

            if retest:
                print("I have seen this controller before, and I'm retesting it either way.")

        if to_append_to_X is None:
            print(f"Deploying the controller {next_controller}")
            performance, behavior = deploy(next_controller)

            to_append_to_X = behavior
            to_append_to_Y = performance
            recorded_perfs[next_centroid] = performance
            recorded_behaviors[next_centroid] = behavior

            if best_performance < performance:
                best_performance = performance
                best_controller = next_controller
                print(f"New best (real) performance found: {performance}")
                print(f"Associated controller: {best_controller}")
                print(f"Associated behavior: {behavior}")

            tested_centroids[next_centroid] += 1
            print(f"Performance of that controller: {performance}")

            dimension = len(behavior)
            # print(dimension)
            # print(f"dimension")
            # _ = input("Press enter to continue.")

        kernel = GPy.kern.Matern52(input_dim=dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension, np.sqrt(0.1))

        X.append(to_append_to_X)
        Y.append(to_append_to_Y - perf_map[next_centroid])

        print(f"X: {X}, Y: {Y}")
        m = GPy.models.GPRegression(
            np.array(X),
            np.array([Y]).T,
            kernel
        )

        # If this is the only use of behaviors_map, then it should just be a np.array.
        # mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        real_map, variance_map = update_real_and_variance_maps(m, perf_map, behaviors_map)
        # print(f"New real map: {real_map}.")

        # Saving the update for visualization
        with open(f"./update_{comment}_{updates}.json", "w") as fp:
            json.dump(
                to_json_writable(real_map),
                fp
            )

        with open(f"./update_metadata_{comment}_{updates}.json", "w") as fp:
            json.dump(
                {
                    "centroid_tested": next_centroid,
                    "associated_controller": next_controller,
                    "recorded_behavior": [float(x) for x in to_append_to_X],
                    "recorded_performance": float(to_append_to_Y),
                    "update": updates
                },
                fp
            )

        # Check stopping conditions
        stopping_condition = check_stopping_condition(real_map, recorded_perfs, alpha)
        if stopping_condition:
            print("The stopping condition has been achieved. Stopping.")
            break
        if updates >= max_iterations:
            break

        updates += 1
        # print("-"*80 + "\n")

    # TODO: return the new best performing controller.
    print(f"I'm out of the main loop. Here's the next best performing controller: {best_controller}")
    return best_controller
