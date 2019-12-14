'''
This script contains the ITaE class, that maintains the map and updates
of an ITaE application.

TODO: test everything.
'''
import numpy as np
import json
import GPy
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
    '''
    def __init__(self, simulate=None, acquisition=None, run=None, random_solution=None, random_selection=None, random_variation=None):
        if simulate is None:
            raise ValueError("The simulate argument is required and mustn't be None.")

        if acquisition is None:
            raise ValueError("The acquisition argument is required and mustn't be None.")

        if run is None:
            raise ValueError("The run argument is required and mustn't be None.")

        # Required fields
        self.simulate = simulate
        self.acquisition = acquisition
        self.run = run

        # Others (for MAP-Elites running, for example)
        self.random_solution = random_solution
        self.random_selection = random_selection
        self.random_variation = random_variation
        self.partition = None
        self.map_elites = None
        self.cells = None
        self.solutions = None
        self.behaviors = None
        self.map_performances = None

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
            self.map_performances[centroid] = cell['performance']

    def get_next_genotype(self, init=False, kappa=0.1):
        # GP = []
        # for i in range(0,len(mu_map)):
        #     GP.append(mu_map[i] + kappa*sigma_map[i])
        # return np.argmax(GP)
        if not init:
            # This won't work, because I'm storing everything in dicts.
            # TODO: fix this.

            return np.argmax(self.real_map + kappa*self.variance)
        if init:
            best_performance = -np.Inf
            best_centroid = None
            for centroid, performance in self.performances: # TODO: implement genotype getting.
                if performance >= best_performance:
                    best_centroid = centroid

            return self.genotypes[best_centroid]
            
        
    def deploy2(self, run):
        '''
        This function takes a run procedure (a function similar to that of
        self.simulation, which takes a genotype and runs it, getting a behavior
        descriptor and a real performance).

        I should maybe maintain a "real_performance" dict that starts as the "map_performance" but is updated at each iteration. Now that I think about it, it doesn't make sense to build a new thing. This is just the mean of the GP.

        STEPS IN THE ORIGINAL ITE:
        1. Load the map
        2. Filter it and create numpy arrays
        3. Create a copy of the performance map.
            - one as the "real" map, another as a saved copy for adding to the mean.
            - They don't really use the n_fits_real for anything. They overwrite it with mean + map from the get-go.
        4. Get the next "index" to test, which is essentially
           getting the next "controller", or genotype in my
           notation
        5. Initialize the X and Y of the GP.
        6. In the while loop:
            6.1. Define the kernel of the GP (this could be done outside I would argue.)
            6.2. predict the value of the mean in the 
            low-dimensional behavior space (i.e. at the 
            description space.)
            6.3. use this mean to compute the actual real values.
            6.4. Compute the percentages of changed values.
            6.5. Get the next index to test.
            6.6. If this new index has already been tested,
                 register that, prepare to test said index again
                 and up the counter for repetitions (?)

                 They don't actually test the index again.
            6.7. If this new index hasn't been tested, simulate
                 it and record performance and behavior.
            6.8. Update X and Y for the GP with relevant
                 values.
            6.9. Maintain the lists of real performances,
                 tested controllers (genotypes), tested
                 descriptions (behaviors), the number of
                 iterations.

                 Things like the real performance list
                 are only being kept for archiving, they
                 aren't used in the code itself as far as I
                 can tell.
            6.10. Check the stopping condition. Stop if
                  that's the case.
        

        TODO: finish this documentation.
        '''
        # Init
        # TODO: get back to this.
        running = True
        tested_behaviors = None
        Y = None
        while running:
            # Main loop.
            # Get and run the best candidate genotype.
            next_genotype, map_performance = self.get_next_genotype()
            behavior, real_performance = run(next_genotype)

            # Update the X and Y of the GP.
            x = np.array([behavior])
            y = np.array([real_performance - map_performance])
            if tested_behaviors is None:
                tested_behaviors = x
            else:
                tested_behaviors = np.vstack((tested_behaviors, x))
            
            if Y is None:
                Y = y
            else:
                Y = np.vstack(Y, y)

            # Compute and update the GP
            dimension = behavior.shape[0]
            ker = GPy.kern.Matern52(dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension,np.sqrt(0.1))
            m = GPy.models.GPRegression(tested_behaviors, Y, ker)

            # Return the map to normal by predicting and adding.
            # means = []
            variances = []
            real_map = {}
            for centroid, behavior in self.behaviors.items():
                mean, variance = m.predict(np.array([behavior]))
                # means.append(mean)
                variances.append(variance)
                # Remember: mean is real - map performance.
                real_map[centroid] = mean + self.map_performances[centroid]

            # This is a very dumb way of doing this, I wonder if I could improve
            # it or just move to storing the raw np arrays as they do.
            # I now think it makes sense to stick with
            # dicts, and to do it this way, because
            # the optimization is continuous, and
            # updating the map by having memory of pre-
            # vious maps doesn't make sense.
            self.real_map = real_map
            means, variances = np.array(means), np.array(variances)

            # decide if we want to stop.
            # TODO: implement the exit condition.
            exit_condition = False
            if exit_condition:
                running = False


    def deploy(self):
        # Initialization.
        rho = 0.4
        variance_noise_square = 0.001
        tested_behaviors = []
        tested_genotypes = {}
        Y = [] # the difference between real performance and map performance.

        next_best_genotype, map_performance = self.get_next_best_genotype()

        # Loop
        while True:
            # Run the next genotype, if we haven't tested it before (TODO: add prior testing verification)
            behavior, real_performance = self.run(next_best_genotype)
            tested_genotypes[next_best_genotype] = real_performance

            # Update the domain and codomain of the GP. GP is modeling
            # the difference between the real performance and the
            # map's performance.
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
            ker = GPy.kern.Matern52(dimension, lengthscale=rho, ARD=False) + GPy.kern.White(dimension,np.sqrt(variance_noise_square))
            m = GPy.models.GPRegression(tested_behaviors, Y, ker)

            # Predict the map "real - simulation"
            # TODO: fill behaviors.
            means, variances = m.predict(self.behaviors)

            # From this, get the map "real" by adding the simulation
            self.pred_performances = means + self.map_performances

            # Huh, the meaning of map here is different from above. It should be querying
            # the updated map.
            next_best_genotype, map_performance = self.get_next_best_genotype()

            # TODO: add an exit condition.

