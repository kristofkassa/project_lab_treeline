import random
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

    def initializePopulation(self, percentage = 0.004):
        """Randomly occupy a certain percentage of the grid.
        """
        self._strategy.occupied_cells = set(random.sample([(i, j) for i in range(self._strategy.grid_size) for j in range(self._strategy.grid_size)], int(self._strategy.grid_size * self._strategy.grid_size * percentage)))

    def simulatePopularizationWithCallback(self):
        self._strategy.simulatePopularization()
