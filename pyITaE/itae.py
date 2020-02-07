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


class ITAE:
    def __init__(self, path, deploy, max_iterations=100, retest=True, comment=""):
        self.path = path
        self.deploy = deploy
        self.max_iterations = max_iterations
        self.retest = retest
        self.comment = comment

        self.alpha = 0.85
        self.kappa = 0.03

        self.perf_map = None
        self.behaviors_map = None
        self.controllers = None
        self.real_map = None
        self.variance_map = None

        self.recorded_perfs = None
        self.recorded_behaviors = None

        self.tested_centroids = None

        self.best_controller = None
        self.best_performance = None

        self.X = None
        self.Y = None
        self.model = None

    def load_map(self):
        with open(self.path) as fp:
            docs = json.load(fp)

        self.perf_map = {}
        self.behaviors_map = {}
        self.controllers = {}
        for doc in docs.values():
            centroid = tuple(doc["centroid"])
            if doc["performance"] is not None:
                self.perf_map[centroid] = doc["performance"]
                self.behaviors_map[centroid] = doc["features"]
                self.controllers[centroid] = doc["solution"]
        
        self.real_map = self.perf_map.copy()

        self.tested_centroids = defaultdict(int)
        self.recorded_perfs = {}
        self.recorded_behaviors = {}

    def check_stopping_condition(self):
        bound = self.alpha*max(self.real_map.values())
        max_perf = max(self.recorded_perfs.values())
        return max_perf > bound
    
    def get_first_centroid(self):
        current_centroid, current_max = None, -np.Inf
        for centroid, performance in self.real_map.items():
            if performance > current_max:
                current_centroid = centroid
                current_max = performance
                print(f"current centroid: {current_centroid}, performance: {performance}")

        print(f"Next centroid: {current_centroid}.")
        return current_centroid

    def update_real_and_variance_maps(self):
        centroids = []
        behaviors = []
        map_performances = []
        for centroid, behavior in self.behaviors_map.items():
            centroids.append(centroid)
            behaviors.append(behavior)
            performance = self.perf_map[centroid]
            assert performance is not None
            map_performances.append(self.perf_map[centroid])

        behaviors = np.array(behaviors)
        mean, variance = self.model.predict(behaviors)
        real_values = mean.T[0] + np.array(map_performances)
        variance_values = variance.T[0]

        # Updating the real and variance maps.
        real_map = {}
        variance_map = {}
        for i, centroid in enumerate(centroids):
            real_map[centroid] = real_values[i]
            variance_map[centroid] = variance_values[i]

        self.real_map = real_map
        self.variance_map = variance_map
    
    def acquisition(self):
        if self.variance_map == None:
            return self.get_first_centroid()
        
        next_centroid, best_bound = None, -np.Inf
        for centroid in self.real_map:
            bound = self.real_map[centroid] + self.kappa * self.variance_map[centroid]
            if bound > best_bound:
                next_centroid = centroid
                best_bound = bound
        print(f"Next centroid: {next_centroid}, value: {best_bound}")
        return next_centroid

    def step(self, update_it=0):
        to_append_to_X = None
        to_append_to_Y = None

        # Get the next controller to test, check if it
        # has been tested in the past.
        next_centroid = self.acquisition()
        next_controller = self.controllers[next_centroid]

        if next_centroid in self.tested_centroids:
            if not self.retest:
                print("I have seen this controller before and I'm not retesting it. I'm assuming it will have the same real performance.")
                print(f"Using previous behavior: {self.behaviors_map[next_centroid]}")
                print(f"Using previous recorded performance: {self.recorded_perfs[next_centroid]}")
                to_append_to_X = self.recorded_behaviors[next_centroid]
                to_append_to_Y = self.recorded_perfs[next_centroid]
                dimension = len(to_append_to_X)

            if self.retest:
                print("I have seen this controller before, and I'm retesting it either way.")

        if to_append_to_X is None:
            print(f"Deploying the controller {next_controller}")
            performance, behavior = self.deploy(next_controller)

            to_append_to_X = behavior
            to_append_to_Y = performance
            self.recorded_perfs[next_centroid] = performance
            self.recorded_behaviors[next_centroid] = behavior

            if self.best_performance < performance:
                self.best_performance = performance
                self.best_controller = next_controller
                print(f"New best (real) performance found: {performance}")
                print(f"Associated controller: {self.best_controller}")
                print(f"Associated behavior: {behavior}")

            self.tested_centroids[next_centroid] += 1
            print(f"Performance of that controller: {performance}")

            dimension = len(behavior)
            # print(dimension)
            # print(f"dimension")
            # _ = input("Press enter to continue.")

        kernel = GPy.kern.Matern52(input_dim=dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension, np.sqrt(0.1))

        self.X.append(to_append_to_X)
        self.Y.append(to_append_to_Y - self.perf_map[next_centroid])

        print(f"X: {self.X}, Y: {self.Y}")
        self.model = GPy.models.GPRegression(
            np.array(self.X),
            np.array([self.Y]).T,
            kernel
        )

        # If this is the only use of behaviors_map, then it should just be a np.array.
        # mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        self.update_real_and_variance_maps()
        # print(f"New real map: {real_map}.")

        # Saving the update for visualization
        with open(f"./update_{self.comment}_{update_it}.json", "w") as fp:
            json.dump(
                to_json_writable(self.real_map),
                fp
            )

        with open(f"./update_metadata_{self.comment}_{update_it}.json", "w") as fp:
            json.dump(
                {
                    "centroid_tested": next_centroid,
                    "associated_controller": next_controller,
                    "recorded_behavior": [float(x) for x in to_append_to_X],
                    "recorded_performance": float(to_append_to_Y),
                    "update": update_it
                },
                fp
            )

    def run(self):
        self.load_map()

        self.X = []
        self.Y = []

        self.best_controller, self.best_performance = None, -np.Inf

        # The main loop
        update_it = 0
        while True:
            # Run a step of the updating
            print("-"*80)
            print(" " * 30 + f"Update {update_it}" + " " * 30)
            self.step(update_it)

            # Check stopping conditions
            stopping_condition = self.check_stopping_condition()
            if stopping_condition:
                print("The stopping condition has been achieved. Stopping.")
                break
            if update_it >= self.max_iterations:
                break

            update_it += 1
            # print("-"*80 + "\n")

        # TODO: return the new best performing controller.
        print(f"I'm out of the main loop. Here's the next best performing controller: {self.best_controller}")

