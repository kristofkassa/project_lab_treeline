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
        gradient = np.arange(self.grid_size) / self.grid_size
        
        self.changes = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):

                # Calculate the gradient value for this cell based on its position
                gradient = 1 - (i / self.grid_size)
                # Adjust the colonization and extinction probabilities based on the gradient value
                c_prob = self.c * (1 - gradient)
                e_prob = self.e / (1 - gradient + 0.001)

                # Check if the current cell has a neighboring cell that is occupied
                if any((i+di, j+dj) in self.occupied_cells for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]):
                    # Determine whether the current cell becomes occupied or remains occupied
                    if random.random() < c_prob and (i, j) not in self.occupied_cells:
                        self.changes.add((i, j))
                    elif random.random() < e_prob and (i, j) in self.occupied_cells:
                        self.changes.add((i, j))
        if not self.changes:
            return
        
        # Apply the changes to the grid
        for i, j in self.changes:
            if (i, j) in self.occupied_cells:
                self.occupied_cells.remove((i, j))
            else:
                self.occupied_cells.add((i, j))


