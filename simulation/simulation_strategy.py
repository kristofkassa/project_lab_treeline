from abc import abstractmethod
import numpy as np
import sys

class SimulationStrategy:

    def __init__(self):
        sys.setrecursionlimit(20000)
        self.grid_size = 150
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.15
        self.c = 0.8
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    @abstractmethod
    def simulatePopularization(self):
        pass

    def _get_neighbors(self, i, j):
        neighbors = []
        for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                neighbors.append((x, y))
        return neighbors

    def _dfs(self, i, j, visited, cluster):
        visited.add((i, j))
        cluster.add((i, j))
        for x, y in self._get_neighbors(i, j):
            if self.occupied_cells[x, y] and (x, y) not in visited:
                self._dfs(x, y, visited, cluster)

    def identifyPercolationClusters(self):
        visited = set()
        clusters = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i, j] and (i, j) not in visited:
                    cluster = set()
                    self._dfs(i, j, visited, cluster)
                    clusters.append(cluster)

        if not clusters:
            self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        else:
            largest_cluster = max(clusters, key=len)
            self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for i, j in largest_cluster:
                self.cluster[i, j] = True

    def markHull(self):
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        visited = set()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cluster[i, j] and (i, j) not in visited:
                    visited.add((i, j))
                    neighbors = self._get_neighbors(i, j)

                    for x, y in neighbors:
                        if not self.cluster[x, y]:
                            self.hull[i, j] = True
                            break
