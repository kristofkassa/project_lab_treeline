import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class GradientRandomMapSimulationStrategy(SimulationStrategy):
    """
    Gradient Random Map simulation strategy.
    """

    def __init__(self):
        super().__init__()
        self._ = 0
        self.neighbors = np.empty_like(self.occupied_cells, dtype=tuple)
        n = self.grid_size
        for i in range(1, n-1):
            for j in range(n):
                self.neighbors[i][j] = (
                    ((i-1)%n, j),
                    (i, (j+1)%n),
                    ((i+1)%n, j),
                    (i, (j-1)%n)
                )
        #copy the first and last column next to themselves for neighbor counting
        for j in range(n):
            self.neighbors[0][j] = (
                (0, j),   #itself
                (0, (j+1)%n),
                (1, j),
                (0, (j-1)%n)
            )
            self.neighbors[n-1][j] = (
                (n-2, j),
                (n-1, (j+1)%n),
                (n-1, j),  #itself
                (n-1, (j-1)%n)
            )

    def simulatePopularization(self):
        # Pre-calculate the gradient values for each cell
        n = self.grid_size
        p_max = 0.8
        gradient_values = np.arange(n) * ((10*p_max-6)/(5*n-10)) + ((6-5*p_max)*n - 6)/(5*n-10)

        self.changes.clear()
        if self._ == 0: #if it is the 0th step (i.e. before the 1st step)
            #clear out initial seeds
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.occupied_cells[i,j] == 1:
                        self.occupied_cells[i,j] = 0
                        self.changes.add((i,j))
            self._ += 1
        elif self._ == 1: #if it is the 1st (and only) step
            for i in range(self.grid_size):
               #prob = self.c * gradient_values[i]
                prob = 1 * gradient_values[i]
                for j in range(self.grid_size):
                    random_number = np.random.rand() #random number from [0.0, 1.0)

                    if random_number < prob:
                        self.occupied_cells[i,j] = 1
                        self.changes.add((i,j))
            self._ += 1
        else: #for further steps
            pass #there is no further step