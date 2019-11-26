'''
This script contains the ITaE class, that maintains the map and updates
of an ITaE application.

TODO: test everything.
'''
import json
from pymelites.map_elites import Cell, MAP_Elites

class ITaE:
    '''
    This class maintains all the relevant information that is processed by the ITaE
    algorithm.

    It takes:
        - The ingredients for the map creation process using MAP_Elites from pymelites.
            - random_solution
            - random_selection
            - random_variation
            - simulate
    It creates empty versions of
        - The partition that describes the map.
        - a map_elites attribute that stores the MAP_Elites object when
          creating the map from scratch.
        - cells and solutions, which essentially constitute the map.
    '''
    def __init__(self, random_solution, random_selection, random_variation, simulate):
        self.random_solution = random_solution
        self.random_selection = random_selection
        self.random_variation = random_variation
        self.simulate = simulate
        self.partition = None
        self.map_elites = None
        self.cells = None
        self.solutions = None

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

        self.cells = {}
        self.solutions = {}
        for cell in doc.values():
            centroid = tuple(cell['centroid'])
            self.cells[centroid] = Cell.from_dict(cell)
            self.solutions[centroid] = cell['solution']
