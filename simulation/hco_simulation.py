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

        self.occupied_and_neighboring_cell_indices = self.update_occupied_and_neighboring_cells()

    def update_occupied_and_neighboring_cells(self):
        occupied_and_neighboring_cell_indices = set(tuple(idx) for idx in np.array(np.where(self.occupied_cells)).T)
        for cell in occupied_and_neighboring_cell_indices.copy():
            occupied_and_neighboring_cell_indices.update(self.neighbors[cell])
        return list(occupied_and_neighboring_cell_indices)

    def simulatePopularization(self):
        self.changes.clear()
        for idx in range(len(self.occupied_and_neighboring_cell_indices)):
            # Sample a random cell from the list of occupied cells and their neighbors
            rand_idx = np.random.randint(len(self.occupied_and_neighboring_cell_indices))
            rand_cell = self.occupied_and_neighboring_cell_indices[rand_idx]
            random_number = np.random.rand()

            if self.occupied_cells[rand_cell]:
                if random_number < self.e:
                    self.occupied_cells[rand_cell] = 0
                    self.changes.add(rand_cell)
            else:
                # Count the number of occupied neighbors of the random cell
                loc_neigh = self.neighbors[rand_cell]
                k = 0
                for neigh in loc_neigh:
                    k += self.occupied_cells[neigh]

                if random_number < self.c * k/4:
                    self.occupied_cells[rand_cell] = 1
                    self.changes.add(rand_cell)

        # Update the list of occupied cells and their neighbors after the changes
        self.occupied_and_neighboring_cell_indices = self.update_occupied_and_neighboring_cells()
