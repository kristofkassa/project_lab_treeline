from abc import abstractmethod
import random
import numpy as np

class SimulationStrategy:

    def __init__(self):
        self.grid_size = 100
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.15
        self.c = 0.2
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    @abstractmethod
    def simulatePopularization(self):
        pass

    def identifyPercolationClusters(self):
        # TODO: implement method
        
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i, j]:
                    if random.random() > 0.5:
                        self.cluster[i, j] = True 

    def markHull(self):
        # TODO: implement method

        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # starting position
        occupied_indices = np.argwhere(self.occupied_cells)
        idx = np.random.choice(len(occupied_indices))
        pos_x, pos_y = occupied_indices[idx]
        self.hull[pos_x, pos_y] = True
        
        for step in range(500):
            # choose a random direction
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            
            # update position based on direction
            if direction == 'up':
                pos_x -= 1
            elif direction == 'down':
                pos_x += 1
            #elif direction == 'left':
            #    pos_y -= 1
            elif direction == 'right':
                pos_y += 1
            
            # check if new position is within bounds
            if pos_x < 0 or pos_x >= self.grid_size or pos_y < 0 or pos_y >= self.grid_size:
                # if new position is out of bounds, wrap around to other side of grid
                pos_x = pos_x % self.grid_size
                pos_y = pos_y % self.grid_size
            
            if self.occupied_cells[pos_x, pos_y]:
                self.hull[pos_x, pos_y] = True

