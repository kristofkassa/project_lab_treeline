import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class HomogeneousContactProcessSimulationStrategy(SimulationStrategy):
    """
    Homogeneous Contact Process simulation strategy.
    """

    def __init__(self):
        super().__init__()
        self.neighbors = np.empty_like(self.occupied_cells, dtype=tuple)
        n = self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.neighbors[i][j] = (
                    ((i+1)%n, j),
                    (i, (j+1)%n),
                    ((i-1)%n, j),
                    (i, (j-1)%n)
                )

    def simulatePopularization(self):
        self.changes.clear()
        for idx in range(self.grid_size**2):
            #Sample a random cell
            rand_cell = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            random_number = np.random.rand()

            if not self.occupied_cells[rand_cell]:
                #Count the number of occupied neighbors of the random cell
                loc_neigh = self.neighbors[rand_cell]
                k = 0
                for neigh in loc_neigh:
                    k += self.occupied_cells[neigh]
                
                if random_number < self.c * k/4:
                    self.occupied_cells[rand_cell] = 1
                    self.changes.add(rand_cell)
            elif random_number < self.e:
                self.occupied_cells[rand_cell] = 0
                self.changes.add(rand_cell)