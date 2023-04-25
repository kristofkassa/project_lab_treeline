import random
import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class GradientContactProcessSimulationStrategy(SimulationStrategy):
    """
    Gradient Random Process simulation strategy.
    """

    def __init__(self):
        super().__init__()

    def simulatePopularization(self):
        # Pre-calculate the gradient values for each cell
        gradient_values = np.arange(self.grid_size) / self.grid_size
        gradient_values = gradient_values.reshape(-1, 1)
        
        self.changes = np.zeros((self.grid_size, self.grid_size)) #set()
        mask = np.pad(self.occupied_cells, ((1, 1), (1, 1)), mode='constant')
        neighbors = (
            mask[:-2, 1:-1]
            + mask[2:, 1:-1]
            + mask[1:-1, :-2]
            + mask[1:-1, 2:]
        )
        neighbors = neighbors > 0
        random_numbers = np.random.rand(self.grid_size, self.grid_size)
        # Adjust the colonization and extinction probabilities based on the gradient value
        c_prob = self.c * gradient_values
        e_prob = self.e / (gradient_values + 0.001)
        # Store the changes
        self.changes[neighbors & (np.less(random_numbers, c_prob)) & (self.occupied_cells != 1)] = 1
        self.changes[neighbors & (np.less(random_numbers, e_prob)) & (self.occupied_cells == 1)] = 1
        # Apply the changes to the grid
        self.occupied_cells = (self.occupied_cells + self.changes) % 2
        
        # Recast self.changes as set()
        changes = np.nonzero(self.changes)
        self.changes = set(list(zip(changes[0], changes[1])))