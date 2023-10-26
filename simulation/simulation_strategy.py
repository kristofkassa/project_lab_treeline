from abc import abstractmethod
import numpy as np
import sys
import math

class SimulationStrategy:

    def __init__(self):
        sys.setrecursionlimit(20000)
        self.grid_size = 200
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.2
        self.c = 0.8
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.occupied_and_neighboring_cell_indices = self.update_occupied_and_neighboring_cells()

    def update_occupied_and_neighboring_cells(self):
        occupied_and_neighboring_cell_indices = set(tuple(idx) for idx in np.array(np.where(self.occupied_cells)).T)
        for cell in occupied_and_neighboring_cell_indices.copy():
            occupied_and_neighboring_cell_indices.update(self.neighbors[cell])
        return list(occupied_and_neighboring_cell_indices)

    @abstractmethod
    def simulatePopularization(self):
        pass

    @abstractmethod
    def nextImage(self):
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
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
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

    def _next_edge_node(self, i, j, prev_i, prev_j):
        directions = [
            (i - 1, j),
            (i, j + 1),
            (i + 1, j),
            (i, j - 1),
        ]

        # Try to find the index of the previous node; if not found, start from 0
        try:
            start_idx = directions.index((prev_i, prev_j))
        except ValueError:
            start_idx = 0

        for d in range(1, 5):  # iterate over the 4 possible directions
            next_i, next_j = directions[(start_idx + d) % 4]
            if 0 <= next_i < self.grid_size and 0 <= next_j < self.grid_size:  # Check bounds
                if self.cluster[next_i, next_j]:
                    return (next_i, next_j)



    def markHull(self):
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Find an initial edge node arbitrarily
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                if self.cluster[i, j]:
                    start_i, start_j = i, j
                    break
            else:
                continue
            break

        # Find the last edge node arbitrarily
        for j in reversed(range(self.grid_size)):
            for i in range(self.grid_size):
                if self.cluster[i, j]:
                    end_i, end_j = i, j
                    break
            else:
                continue
            break

        print(start_i, start_j, end_i, end_j)

        # Start edge following
        prev_i, prev_j = start_i, start_j
        curr_i, curr_j = self._next_edge_node(start_i, start_j, start_i, start_j)
        self.hull[start_i, start_j] = True

        while curr_i != end_i or curr_j != end_j:
            self.hull[curr_i, curr_j] = True
            next_i, next_j = self._next_edge_node(curr_i, curr_j, prev_i, prev_j)
            prev_i, prev_j = curr_i, curr_j
            curr_i, curr_j = next_i, next_j

        self.hull[curr_i, curr_j] = True  # Mark the starting node again to close the hull
        return self.calculate_fractal_dimension_boxcounting(), self.calculate_fractal_dimension_correlation()

    def calculate_fractal_dimension_boxcounting(self):
        """
            Initialize the box sizes and counts.
            Iterate through the different box sizes.
            For each box size, cover the hull with non-overlapping boxes.
            Count the number of boxes that intersect with the hull.
            Calculate the slope of the log-log plot of box size versus the count of boxes that intersect the hull.
        """
        box_sizes = [2 ** i for i in range(int(math.log2(self.grid_size)) + 1)]
        box_counts = []

        for box_size in box_sizes:
            count = 0
            for i in range(0, self.grid_size, box_size):
                for j in range(0, self.grid_size, box_size):
                    for x in range(i, min(i + box_size, self.grid_size)):
                        for y in range(j, min(j + box_size, self.grid_size)):
                            if self.hull[x, y]:
                                count += 1
                                break
                        else:
                            continue
                        break
            box_counts.append(count)

        log_box_sizes = [math.log(size) for size in box_sizes]
        log_box_counts = [math.log(count) for count in box_counts]

        # Calculate the slope of the log-log plot using linear regression
        slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)
        fractal_dimension = -slope
        print("Fractal dimension:", fractal_dimension)
        return fractal_dimension
    
    def calculate_fractal_dimension_correlation(self):
        """
        Calculate the fractal dimension using the correlation dimension method.
        This method computes the correlation sum C(r) which counts the number of point pairs that have distance less than r.
        Then it estimates the dimension as the slope of log(C(r)) vs log(r).
        """
        max_radius = self.grid_size // 2
        radius_sizes = [2 ** i for i in range(int(math.log2(max_radius)) + 1)]
        correlation_sums = []

        occupied_coordinates = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.hull[x, y]]
        num_points = len(occupied_coordinates)

        for radius in radius_sizes:
            correlation_sum = 0
            for i in range(num_points):
                for j in range(i+1, num_points):
                    x1, y1 = occupied_coordinates[i]
                    x2, y2 = occupied_coordinates[j]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance <= radius:
                        correlation_sum += 1
            correlation_sums.append(correlation_sum)

        log_radius_sizes = [math.log(size) for size in radius_sizes]
        log_correlation_sums = [math.log(correlation_sum) for correlation_sum in correlation_sums]

        # Calculate the slope of the log-log plot using linear regression
        slope, _ = np.polyfit(log_radius_sizes, log_correlation_sums, 1)
        fractal_dimension = slope
        print("Fractal dimension correlation:", fractal_dimension)
        return fractal_dimension
