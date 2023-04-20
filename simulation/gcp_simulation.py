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
        
        self.changes = set()
        mask = np.pad(self.occupied_cells, ((1, 1), (1, 1)), mode='constant')
        neighbors = (
            mask[:-2, 1:-1]
            + mask[2:, 1:-1]
            + mask[1:-1, :-2]
            + mask[1:-1, 2:]
        )
        neighbors = neighbors > 0
        random_numbers = np.random.rand(self.grid_size, self.grid_size)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate the gradient value for this cell based on its position
                gradient = 1 - gradient_values[i]
                # Adjust the colonization and extinction probabilities based on the gradient value
                c_prob = self.c * (1 - gradient)
                e_prob = self.e / (1 - gradient + 0.001)

                if neighbors[i, j]:
                    if random_numbers[i, j] < c_prob and not self.occupied_cells[i, j]:
                        self.changes.add((i, j))
                    elif random_numbers[i, j] < e_prob and self.occupied_cells[i, j]:
                        self.changes.add((i, j))
                        
        if not self.changes:
            return

        # Apply the changes to the grid
        for i, j in self.changes:
            self.occupied_cells[i, j] = not self.occupied_cells[i, j]

