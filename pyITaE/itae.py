import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from collections import defaultdict

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def to_json_writable(dict_):
    """
    TODO: embetter this function
    """
    new_dict = {
        str(k): v for k, v in dict_.items()
    }
    return new_dict

class ITAE:
    def __init__(self, path, deploy, max_iterations=100, kernel=None, retest=True, goal=None, distance_to_goal=None, comment="", path_to_updates=".", performance_bound=None):
        """
        ITAE class: an object that runs the Intelligent Trial and Error
        algorithm, maintaining the results of the deployment and the current
        model of the world.

        It takes:
            path (str): a path to the prior json file outputted by pymelites.
            deploy (f): a deploy function that takes a genotype (or controller) and 
                    returns a tuple (performance, behavior). If you're familiar
                    with pymelites, think of the "simulate" function in the
                    MAP-Elites object.
        Optional parameters:
            max_iterations (int): maximum iterations of Bayesian Updating
                                  at run time.
            kernel (sci-kit learn's): the kernel for the GP.
            retest (bool): a boolean flag that says whether or not to retest
                           a controller.
            goal (float): a target performance (if we shouldn't maximize the
                          performance itself)
            distance_to_goal (float): Stopping criteria if goal is not None,
                we will stop if we are this close to the goal.
            comment (str): a string that will be included in the files' names.
            path_to_updates (str): where to save the updates.
            performance_bound: a lower bound on the performance that changes
              the stopping criteria to be "finding a level of at least
              {performance_bound} performance".
        """
        self.path = path
        self.deploy = deploy
        self.max_iterations = max_iterations
        self.retest = retest
        self.comment = comment
        self.path_to_updates = path_to_updates
        self.performance_bound = performance_bound

        if kernel is None:
            """
            Default kernel: Matern 5/2 plus diagonal noise.
            """
            kernel = 1 * Matern(nu=5/2) + WhiteKernel(noise_level=np.log(2))

        self.kernel = kernel

        # For bent acquisitions
        if goal is not None:
            assert isinstance(goal, (float, int)), "goal should be a float or int."
            assert isinstance(distance_to_goal, (float, int)), "if you specified goal, you need to specify distance to goal for the stopping condition"
        self.goal = goal
        self.distance_to_goal = distance_to_goal

        self.alpha = 0.85 # percentage of performance that we're happy with.
        self.kappa = 0.03 # UCB constant.

        self.perf_map = None
        self.behaviors_map = None
        self.controllers = None
        self.real_map = None
        self.sigma_map = None

        self.recorded_perfs = None
        self.recorded_behaviors = None

        self.tested_centroids = None

        self.best_controller = None
        self.best_performance = None

        self.X = None
        self.Y = None
        self.model = None

        self.update_it = None

    def _objective(self, r):
        """
        This function returns the objective function that
        we are trying to optimize. If the self.goal is None,
        then it is performance (and we'll return the identity
        function), but if self.goal is a numerical value, we
        return
        
            - (r - self.goal) ** 2

        This way, we can optimize for self.goal instead of
        maximizing performance.
        """
        if self.goal is None:
            return r
        else:
            return - (r - self.goal) ** 2

    def update_generation(self):
        """
        This function creates a new generation-like file
        with the performances stored in real_map.

        The input follows the name conventions of the run method of
        the ITAE class, that is: path is a string with the filepath to
        the generation string, and real_map is a dict {centroid: performance}.
        """
        with open(self.path) as fp:
            generation = json.load(fp)

        for centroid, real_perf in self.real_map.items():
            key = str(centroid).replace("(", "[").replace(")", "]")
            generation[key]["performance"] = real_perf

        path_to_update = f"{self.path_to_updates}/update_{self.comment}_{self.update_it}.json"
        with open(path_to_update, "w") as fp:
            json.dump(generation, fp)

    def load_map(self):
        with open(self.path) as fp:
            docs = json.load(fp)

        self.perf_map = {}
        self.behaviors_map = {}
        self.controllers = {}
        keys = None
        for doc in docs.values():
            centroid = tuple(doc["centroid"])
            if doc["performance"] is not None:
                self.perf_map[centroid] = doc["performance"]
                if isinstance(doc["features"], dict):
                    if keys is None:
                        keys = list(doc["features"].keys())
                        keys.sort()

                    features_as_list = [doc["features"][k] for k in keys]
                    self.behaviors_map[centroid] = features_as_list
                else:
                    self.behaviors_map[centroid] = doc["features"]
                self.controllers[centroid] = doc["solution"]
        
        self.real_map = self.perf_map.copy()

        self.tested_centroids = defaultdict(int)
        self.recorded_perfs = {}
        self.recorded_behaviors = {}

    def check_stopping_condition(self):
        if self.update_it >= self.max_iterations:
            return True

        if self.goal is not None:
            """
            If we're not maximizing, we will need to use
            self.distance_to_goal to stop.
            """
            return np.abs(self.best_performance - self.goal) < self.distance_to_goal
        elif self.performance_bound is not None:
            """
            This is an alternative stopping condition to the original
            one. In this version, the algorithm only stops after finding
            a "good enough" controller, where by good enough we mean
            >= self.performance_bound.
            """
            return self.best_performance >= self.performance_bound
        else:
            """
            This stopping condition is the one in the original ITAE
            algorithm implementation. We have found that this one
            only works when the prior maps are very illuminated.
            """
            bound = self.alpha*max(self.real_map.values())
            max_perf = max(self.recorded_perfs.values())
            return max_perf > bound

    def get_first_centroid(self):
        current_centroid, current_max = None, -np.Inf
        for centroid, performance in self.real_map.items():
            if performance > current_max:
                current_centroid = centroid
                current_max = self._objective(performance)
                # print(f"current centroid: {current_centroid}, performance: {performance}")

        # print(f"Next centroid: {current_centroid}.")
        return current_centroid

    def update_real_and_sigma_maps(self):
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
        mean, sigma = self.model.predict(behaviors, return_std=True)
        # print(f"mean: {mean}")
        real_values = mean.T + np.array(map_performances)
        sigma_values = sigma

        # Updating the real and variance maps.
        real_map = {}
        sigma_map = {}
        # print(f"real_values: {real_values}")
        # print(f"sigma_values: {sigma_values}")
        for i, centroid in enumerate(centroids):
            real_map[centroid] = real_values[i]
            sigma_map[centroid] = sigma_values[i]

        self.real_map = real_map
        self.sigma_map = sigma_map

    def acquisition(self):
        if self.sigma_map == None:
            next_centroid = self.get_first_centroid()
            print(f"First centroid to test: {next_centroid}")
            print(f"Its performance in the prior: {self.perf_map[next_centroid]}")
            return self.get_first_centroid()
        
        if self.retest:
            centroids = self.real_map.keys()
        else:
            centroids = set(self.real_map.keys()) - set(self.tested_centroids.keys())
            centroids = list(centroids)

        next_centroid, best_bound = None, -np.Inf
        for centroid in centroids:
            bound = self.real_map[centroid] + self.kappa * self.sigma_map[centroid]
            bound = self._objective(bound)
            if bound > best_bound:
                next_centroid = centroid
                best_bound = bound
        print(f"Next centroid: {next_centroid}, bound: {best_bound}")
        print(f"Our current prediction: {self.real_map[next_centroid]}")
        return next_centroid

    def step(self):
        to_append_to_X = None
        almost_to_append_to_Y = None

        # Get the next controller to test, check if it
        # has been tested in the past.
        next_centroid = self.acquisition()
        next_controller = self.controllers[next_centroid]

        if to_append_to_X is None:
            print(f"Deploying the controller {next_controller}")
            tuple_ = self.deploy(next_controller)
            if len(tuple_) == 2:
                performance, behavior = tuple_
                metadata = None
            elif len(tuple_) == 3:
                performance, behavior, metadata = tuple_
            # TODO: I need to change the behavior here
            if isinstance(behavior, dict):
                keys = list(behavior.keys())
                keys.sort()
                behavior_as_list = [behavior[k] for k in keys]
                to_append_to_X = behavior_as_list
            else:
                to_append_to_X = list(behavior)
            almost_to_append_to_Y = performance
            self.recorded_perfs[next_centroid] = performance
            self.recorded_behaviors[next_centroid] = behavior

            best_obj_value = self._objective(self.best_performance)
            current_obj_value = self._objective(performance)
            if best_obj_value < current_obj_value:
                self.best_performance = performance
                self.best_controller = next_controller
                print(f"New best (real) performance found: {performance}")
                if self.goal is not None:
                    print(f"We're aiming for {self.goal} performance.")
                print(f"Associated controller: {self.best_controller}")
                print(f"Associated behavior: {behavior}")

            self.tested_centroids[next_centroid] += 1
            print(f"Performance of that controller: {performance}")

            dimension = len(behavior)

        self.X.append(to_append_to_X)
        self.Y.append(almost_to_append_to_Y - self.perf_map[next_centroid])

        print(f"X: {self.X}, Y: {self.Y}")
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        # Will it complain about sizes?
        self.model.fit(self.X, self.Y)
        print(f"kernel: {self.model.kernel_}")

        # If this is the only use of behaviors_map, then it should just be a np.array.
        # mean, variance = m.predict(behaviors_map)

        # But I need to find a consistent way of adding them here.
        self.update_real_and_sigma_maps()

        # Saving the update for visualization
        self.update_generation()

        # Saving the update metadata
        path_to_metadata = f"{self.path_to_updates}/update_metadata_{self.comment}_{self.update_it}.json"
        with open(path_to_metadata, "w") as fp:
            json.dump(
                {
                    "centroid_tested": next_centroid,
                    "associated_controller": next_controller,
                    "recorded_behavior": [float(x) for x in to_append_to_X],
                    "recorded_performance": float(almost_to_append_to_Y),
                    "best_controller": self.best_controller,
                    "best_performance": float(self.best_performance),
                    "update": self.update_it,
                    "goal": self.goal,
                    "metadata": metadata
                },
                fp
            )


    def run(self):
        self.load_map()

        self.X = []
        self.Y = []

        self.best_controller, self.best_performance = None, -np.Inf

        # The main loop
        self.update_it = 0
        while True:
            # Run a step of the updating
            print("-"*80)
            print(" " * 30 + f"Update {self.update_it}" + " " * 30)
            self.step()

            # Check stopping conditions
            stopping_condition = self.check_stopping_condition()
            if stopping_condition:
                print("The stopping condition has been achieved. Stopping.")
                break

            if self.update_it >= self.max_iterations - 1:
                break

            self.update_it += 1
            # print("-"*80 + "\n")

        # TODO: return the new best performing controller.
        print(f"Here's the next best performing controller: {self.best_controller}, Here's the best performance: {self.best_performance}")

    def plot_update_surface(self, xlims, ylims):
        """
        TODO: write docstring, add plotting of the variance.

        It seems that I'll only be able to print whatever's on the
        real map, right?, because the model is learning real - simulation, and
        simulation is only defined in some points of the box.
        """
        # The idea: plotting whatever the model's predicting in a bounded box.
        domain_X = np.linspace(xlims[0], xlims[1], 100)
        domain_Y = np.linspace(ylims[0], ylims[1], 100)
        domain_X, domain_Y = np.meshgrid(domain_X, domain_Y)

        Z = np.zeros(shape=domain_X.shape)
        for i, row_X in enumerate(domain_X):
            for j, x in enumerate(row_X):
                y = domain_Y[i, j]
                mean, _ = self.model.predict((x, y))
                Z[i, j] = mean

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(domain_X, domain_Y, Z)