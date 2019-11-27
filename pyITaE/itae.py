'''
This script contains the ITaE class, that maintains the map and updates
of an ITaE application.

TODO: test everything.
'''
import numpy as np
import json
import GPy
# TODO: add different kernels than RBF.
from sklearn.gaussian_process import GaussianProcessRegressor
from pymelites.map_elites import Cell, MAP_Elites

class ITaE:
    '''
    This class maintains all the relevant information that is processed by the ITaE
    algorithm.

    It takes:
        - The ingredients for the map creation process using MAP_Elites from pymelites.
            - simulate
            - acquisition
            - random_solution
            - random_selection
            - random_variation
          Notice that the simulate is first, and acquisition is second.
    It creates empty versions of
        - The partition that describes the map.
        - a map_elites attribute that stores the MAP_Elites object when
          creating the map from scratch.
        - cells and solutions, which essentially constitute the map.
    
    TODO:
        - Implement get_next_best_genotype using the acquisition function.
        - Test the compute_map method.
        - Load behaviors and store them in load_map.
        - Add a run kwarg, with the deployment function.
    '''
    def __init__(self, simulate=None, acquisition=None, random_solution=None, random_selection=None, random_variation=None):
        if simulate is None:
            raise ValueError("The simulate argument is required and mustn't be None.")

        if acquisition is None:
            raise ValueError("The acquisition argument is required and mustn't be None.")

        self.simulate = simulate
        self.acquisition = acquisition

        self.random_solution = random_solution
        self.random_selection = random_selection
        self.random_variation = random_variation
        self.partition = None
        self.map_elites = None
        self.cells = None
        self.solutions = None
        self.behaviors = None

    def compute_map(self, partition, generations, iterations_per_gen, generation_path='.'):
        '''
        This function computes the map according to the given partition (see the MAP_Elites'
        create_cell method), amount of generations, amount of iterations per generation,
        and stores the generations in the given path.

        TODO:
            - add more kwargs, like amount_of_elites in the cell creation.
            - test this.
        '''
        # Create the MAP_Elites object, create the cells, and compute the archive

        if self.random_solution is None:
            raise ValueError("random_solution must be a function, not None.")

        if self.random_selection is None:
            raise ValueError("random_selection must be a function, not None.")

        if self.random_variation is None:
            raise ValueError("random_variation must be a function, not None.")

        self.map_elites = MAP_Elites(
            self.random_solution,
            self.random_selection,
            self.random_variation,
            self.simulate
        )

        # Create the cells
        # TODO: if I were to add amount_of_elites kwarg, this
        # needs to change.
        self.map_elites.create_cells(partition)

        # Compute the map
        self.map_elites.compute_archive(
            generations, iterations_per_gen, generation_path
        )

        # Store cells and solutions (should I do a deep copy of them?)
        self.cells = self.map_elites.cells.copy()
        self.solutions = self.map_elites.solutions.copy()


    def load_map(self, path):
        '''
        This function loads a 'generation_k.json' file, created by pymelites
        after running compute_archive in a MAP_elites object.

        TODO:
            - test this.
        '''
        with open(path) as fp:
            doc = json.load(fp)

        # TODO: load behaviors as well
        self.cells = {}
        self.solutions = {}
        for cell in doc.values():
            centroid = tuple(cell['centroid'])
            self.cells[centroid] = Cell.from_dict(cell)
            self.solutions[centroid] = cell['solution']
        
    
    def deploy(self):
        # Initialization
        rho = 0.4
        variance_noise_square = 0.001
        tested_behaviors = []
        Y = [] # the difference between real performance and map performance.

        next_best_genotype, map_performance = self.get_next_best_genotype()

        # Loop
        while True:
            # This shouldn't be simulate, but run. A new function in deployment.
            behavior, real_performance = self.simulate(next_best_genotype)
            
            # Update the domain and codomain of the GP.
            if len(tested_behaviors) == 0:
                tested_behaviors = np.array(list(behavior))
            else:
                tested_behaviors = np.vstack((tested_behaviors, behavior))
            
            y = real_performance - map_performance
            if len(Y) == 0:
                Y = np.array([y])
            else:
                Y = np.vstack((Y, np.array([y])))
            
            # Compute the GP with the updated domain and codomain.
            dimension = behavior.shape[0]
            ker = GPy.kern.Matern52(dimension,lengthscale = rho, ARD=False) + GPy.kern.White(dimension,np.sqrt(variance_noise_square))
            m = GPy.models.GPRegression(tested_behaviors, Y, ker)

            # Predict the map "real - simulation"
            # TODO: fill behaviors.
            means, variances = m.predict(self.behaviors)

            # From this, get the map "real" by adding the simulation
            self.pred_performances = means + self.map_performances

            next_best_genotype, map_performance = self.get_next_best_genotype()

            # TODO: add an exit condition.

            

