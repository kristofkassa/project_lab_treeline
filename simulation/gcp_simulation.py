import random
import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class GradientContactProcessSimulationStrategy(SimulationStrategy):
    """
    Gradient Random Process simulation strategy.
    """

    def __init__(self):
        super().__init__()
        self.neighbors = np.empty_like(self.occupied_cells, dtype=tuple)
        n = self.grid_size
        for i in range(1, n-1):
            for j in range(n):
                self.neighbors[i][j] = (
                    ((i+1)%n, j),
                    (i, (j+1)%n),
                    ((i-1)%n, j),
                    (i, (j-1)%n)
                )
        #copy the first and last column next to themselves for neighbor counting
        for j in range(n):
            self.neighbors[0][j] = (
                (1, j),
                (0, (j+1)%n),
                (0, j),   #itself
                (0, (j-1)%n)
            )
            self.neighbors[n-1][j] = (
                (n-1, j),  #itself
                (n-1, (j+1)%n),
                (n-2, j),
                (n-1, (j-1)%n)
            )

    def simulatePopularization(self):
        # Pre-calculate the gradient values for each cell
        gradient_values = np.arange(self.grid_size) / self.grid_size

        self.changes.clear()
        for _ in range(self.grid_size**2): #regular Monte Carlo step
            # Sample a random cell from the lattice
            rand_cell = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            random_number = np.random.rand()

            # Adjust the colonization and extinction probabilities based on the gradient value
            i = rand_cell[0]
            c_prob = 1 * gradient_values[i]
            e_prob = self.e #/ (gradient_values[i] + 0.0001)

            if self.occupied_cells[rand_cell]:
                if random_number < e_prob:
                    self.occupied_cells[rand_cell] = 0
                    self.changes.add(rand_cell)
            else:
                # Count the number of occupied neighbors of the random cell
                loc_neigh = self.neighbors[rand_cell]
                k = 0
                for neigh in loc_neigh:
                    k += self.occupied_cells[neigh]

                if random_number < c_prob * k/4:
                    self.occupied_cells[rand_cell] = 1
                    self.changes.add(rand_cell)

        # Update the list of occupied cells and their neighbors after the changes
        self.occupied_and_neighboring_cell_indices = self.update_occupied_and_neighboring_cells()