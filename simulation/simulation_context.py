import numpy as np
from simulation.simulation_strategy import SimulationStrategy

class SimulationContext:

    def __init__(self, strategy: SimulationStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> SimulationStrategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: SimulationStrategy) -> None:
        """
        Context allows replacing a Strategy object at runtime.
        """
        print('Strategy switched')
        self._strategy = strategy


    def setColonization(self, c_str):
        try:
            c = float(c_str)
            if c > 0:
                self._strategy.c = c
        except ValueError:
            pass

    def setExtinction(self, e_str):
        try:
            e = float(e_str)
            if e > 0:
                self._strategy.e = e    
        except ValueError:
            pass   

    def initializePopulation(self, percentage = 0.001):
        """Randomly occupy a certain percentage of the grid.
        """
        n_occupied = int(self._strategy.grid_size * self._strategy.grid_size * percentage)
        indices = np.random.choice(self._strategy.grid_size * self._strategy.grid_size, n_occupied, replace=False)
        row_indices = indices // self._strategy.grid_size
        col_indices = indices % self._strategy.grid_size
        self._strategy.occupied_cells[row_indices, col_indices] = True

    def simulatePopularizationWithCallback(self):
        self._strategy.simulatePopularization()
