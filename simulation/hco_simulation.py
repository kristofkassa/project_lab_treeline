import random
from simulation.simulation_strategy import SimulationStrategy

class HomogeneousContactProcessSimulationStrategy(SimulationStrategy):
    """
    Homogeneous Contact Process simulation strategy.
    """

    def __init__(self):
        super().__init__()

    def simulatePopularization(self):
        self.changes = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Check if the current cell has a neighboring cell that is occupied
                if any((i+di, j+dj) in self.occupied_cells for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]):
                    # Determine whether the current cell becomes occupied or remains occupied
                    if random.random() < self.c and (i, j) not in self.occupied_cells:
                        self.changes.add((i, j))
                    elif random.random() < self.e and (i, j) in self.occupied_cells:
                        self.changes.add((i, j))
        if not self.changes:
            return
        # Apply the changes to the grid
        for i, j in self.changes:
            if (i, j) in self.occupied_cells:
                self.occupied_cells.remove((i, j))
            else:
                self.occupied_cells.add((i, j))