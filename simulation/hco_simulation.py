import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class HomogeneousContactProcessSimulationStrategy(SimulationStrategy):
    """
    Homogeneous Contact Process simulation strategy.
    """

    def __init__(self):
        super().__init__()

    def simulatePopularization(self):
        
        self.changes.clear()
        mask = np.pad(self.occupied_cells_b, ((1, 1), (1, 1)), mode='constant')
        neighbors = (
            mask[:-2, 1:-1]
            + mask[2:, 1:-1]
            + mask[1:-1, :-2]
            + mask[1:-1, 2:]
        )
        neighbors = neighbors > 0
        random_numbers = np.random.rand(*self.occupied_cells_b.shape)
        become_occupied = (random_numbers < self.c) & (~self.occupied_cells_b) & neighbors
        become_unoccupied = (random_numbers < self.e) & self.occupied_cells_b
        self.occupied_cells_b[become_occupied] = True
        self.occupied_cells_b[become_unoccupied] = False

        changed_indices = np.argwhere(become_occupied | become_unoccupied)
        changed_indices_tuples = [tuple(x) for x in changed_indices]

        self.changes.update(changed_indices_tuples)
        n_changes = len(self.changes)
        if not n_changes:
            return
        for i in range(n_changes):
            pos = tuple(next(iter(self.changes)))
            if self.occupied_cells_b[pos]:
                self.occupied_cells_b[pos] = False
            else:
                self.occupied_cells_b[pos] = True
