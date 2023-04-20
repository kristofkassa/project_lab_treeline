from abc import abstractmethod
import numpy as np

class SimulationStrategy:

    def __init__(self):
        self.grid_size = 100
        self.occupied_cells_b = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.15
        self.c = 0.2

    @abstractmethod
    def simulatePopularization(self):
        pass

    def identifyPercolationClusters(self):
        # TODO: implement method
        pass

    def markHull(self):
        # TODO: implement method
        pass

