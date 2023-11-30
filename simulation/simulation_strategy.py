from abc import abstractmethod
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

class SimulationStrategy:

    def __init__(self):
        sys.setrecursionlimit(50000)
        self.grid_size = 2**7
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.2
        self.c = 0.8
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull_list = []

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
        self.hull_list.append((start_i, start_j))

        while curr_i != end_i or curr_j != end_j:
            self.hull[curr_i, curr_j] = True
            self.hull_list.append((curr_i, curr_j))
            next_i, next_j = self._next_edge_node(curr_i, curr_j, prev_i, prev_j)
            prev_i, prev_j = curr_i, curr_j
            curr_i, curr_j = next_i, next_j

        self.hull[curr_i, curr_j] = True  # Mark the starting node again to close the hull
        self.hull_list.append((curr_i, curr_j))

        #print(self.hull_list)
        self.calculate_fractal_dimension_ruler()
        self.calculate_fractal_dimension_avgdist()
        return self.calculate_fractal_dimension_boxcounting(), self.calculate_fractal_dimension_correlation()

    def calculate_fractal_dimension_boxcounting(self):
        """
            Initialize the box sizes and counts.
            Iterate through the different box sizes.
            For each box size, cover the hull with non-overlapping boxes.
            Count the number of boxes that intersect with the hull.
            Calculate the slope of the log-log plot of box size versus the count of boxes that intersect the hull.
        """
        box_sizes = [2 ** i for i in range(0,int(math.log2(self.grid_size)) )]
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

        #plot the data points
        plt.figure(figsize=(8, 6))
        plt.plot(log_box_sizes[1:-1], log_box_counts[1:-1], 'o', color = 'lime')
        plt.plot([log_box_sizes[i] for i in [0,-1]], [log_box_counts[i] for i in [0,-1]], 'o', color = 'gray')
        plt.title('Log-Log Plot of Box Sizes vs. Box Counts')
        plt.xlabel('Log(Box Sizes)')
        plt.ylabel('Log(Box Counts)')

        # Calculate the slope of the log-log plot using linear regression
        slope, intercept = np.polyfit(log_box_sizes[1:-1], log_box_counts[1:-1], 1)
        fractal_dimension = -slope
        print("Box Dimension:", fractal_dimension)


        #plot the actual and desired regression line
        regression_line = slope * np.array(log_box_sizes) + intercept
        magic_line = (-1.75) * np.array(log_box_sizes) + intercept
        plt.plot(log_box_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_box_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()

        return fractal_dimension
    
    def calculate_fractal_dimension_correlation(self):
        """
        Calculate the fractal dimension using the correlation dimension method.
        This method computes the correlation sum C(r) which counts the number of point pairs that have distance less than r.
        Then it estimates the dimension as the slope of log(C(r)) vs log(r).
        """
        max_radius = self.grid_size // 2
        radius_sizes = [2 ** i for i in range(0,int(math.log2(max_radius)))]
        correlation_sums = []

        occupied_coordinates = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.hull[x, y]]
        num_points = len(occupied_coordinates)

        for radius in radius_sizes:
            correlation_sum = 0
            for i in range(num_points):
                for j in range(i+1, num_points):
                    x1, y1 = occupied_coordinates[i]
                    x2, y2 = occupied_coordinates[j]
                    distance = dist(occupied_coordinates[i], occupied_coordinates[j])
                    if distance <= radius:
                        correlation_sum += 1
            correlation_sums.append(correlation_sum)

        log_radius_sizes = [math.log(size) for size in radius_sizes]
        log_correlation_sums = [math.log(correlation_sum) for correlation_sum in correlation_sums]

        #plot the data points
        plt.figure(figsize=(8, 6))
        plt.plot(log_radius_sizes[1:-1], log_correlation_sums[1:-1], 'o', color = 'lime')
        plt.plot([log_radius_sizes[i] for i in [0, -1]], [log_correlation_sums[i] for i in [0,-1]], 'o', color = 'gray')
        plt.title('Log-Log Plot of Radius Sizes vs. Correlation Sums')
        plt.xlabel('Log(Radius Sizes)')
        plt.ylabel('Log(Correlation Sums)')

        # Calculate the slope of the log-log plot using linear regression
        slope, intercept = np.polyfit(log_radius_sizes[1:-1], log_correlation_sums[1:-1], 1)
        fractal_dimension = slope
        print("Correlation Dimenson:", fractal_dimension)


        #plot the actual and desired regression line
        regression_line = slope * np.array(log_radius_sizes) + intercept
        magic_line = (1.75) * np.array(log_radius_sizes) + intercept
        plt.plot(log_radius_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_radius_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()

        return fractal_dimension

    def calculate_fractal_dimension_ruler(self):
        #try:
        max_size = self.grid_size #//4
        ruler_sizes = [2 ** i for i in range(0, int(math.log2(max_size)) )]
        ruler_counts = []

        hull_cells = self.hull_list

        for ruler_size in ruler_sizes:
            _ = 0
            ruler_count = 0
            print("r=", ruler_size)
            cell0 = hull_cells[0]
            i = 1
            while i < len(hull_cells):
                while (i < len(hull_cells)) and (dist(cell0, hull_cells[i]) <= ruler_size):
                    i+=1
                print(cell0)
                cell0 = hull_cells[i-1]
                ruler_count += 1

                _ += 1
                if _ > self.grid_size ** 2: 
                    print("INFINITE LOOP")
                    break #to avoid inifite loops in developement stage
            print(hull_cells[i-1])
            ruler_counts.append(ruler_count)
            print("count = ", ruler_count, "\n")

        

        log_ruler_sizes = [math.log(ruler_size) for ruler_size in ruler_sizes]
        log_ruler_counts = [math.log(ruler_count) for ruler_count in ruler_counts]

        #plot the data points
        plt.figure(figsize=(8, 6))
        plt.plot(log_ruler_sizes[1:-1], log_ruler_counts[1:-1], 'o', color = 'lime')
        plt.plot([log_ruler_sizes[i] for i in [0,-1]], [log_ruler_counts[i] for i in [0,-1]], 'o', color = 'gray')
        plt.title('Log-Log Plot of Ruler Sizes vs. Ruler Counts')
        plt.xlabel('Log(Ruler Sizes)')
        plt.ylabel('Log(Ruler Counts)')

        # Calculate the slope of the log-log plot using linear regression
        slope, intercept = np.polyfit(log_ruler_sizes[1:-1], log_ruler_counts[1:-1], 1)
        fractal_dimension = -slope
        print("Ruler Dimension:", fractal_dimension)


        regression_line = slope * np.array(log_ruler_sizes) + intercept
        magic_line = (-1.75) * np.array(log_ruler_sizes) + intercept
        plt.plot(log_ruler_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_ruler_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()
        return fractal_dimension
       # except Exception as error:
       #     print(error)
       #     return -1

    def calculate_fractal_dimension_avgdist(self):
        k_lengths = [2 ** i for i in range(0, int(math.log2(len(self.hull_list)//2)) )]
        avg_dists = []

        for k in k_lengths:
            m = len(self.hull_list) // k
            avg = 0.0
            for i in range(1, m):
                print("k=",k,", i*k=",i*k,", m=",m)
                avg += dist(self.hull_list[i*k], self.hull_list[(i-1)*k])
            avg /= m
            print("k=", k, "  avg=", avg)

            avg_dists.append(avg)

        log_k_lengths = [math.log(k) for k in k_lengths]
        log_avg_dists = [math.log(avg) for avg in avg_dists]

        #plot the data points
        plt.figure(figsize=(8, 6))
        plt.plot(log_k_lengths[2:-2], log_avg_dists[2:-2], 'o', color = 'lime')
        plt.plot([log_k_lengths[i] for i in [0,1,-2,-1]], [log_avg_dists[i] for i in [0,1,-2,-1]], 'o', color = 'gray')
        plt.title('Log-Log Plot of k values vs. Average Distances')
        plt.xlabel('Log(k)')
        plt.ylabel('Log(Average Distances)')

        # Calculate the slope of the log-log plot using linear regression
        slope, intercept = np.polyfit(log_k_lengths[2:-2], log_avg_dists[2:-2], 1)
        fractal_dimension = 1/slope
        print("AvgDist Dimension:", fractal_dimension)


        regression_line = slope * np.array(log_k_lengths) + intercept
        magic_line = (1/1.75) * np.array(log_k_lengths) + intercept
        plt.plot(log_k_lengths, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = 1/{fractal_dimension:.2f})')
        plt.plot(log_k_lengths, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = 1/1.75)')

        plt.legend()
        plt.grid(True)
        plt.show()
        return fractal_dimension

    def topleft_cell_of_hull(self):
        hull_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.hull[x, y]]
        hull_y = [cell[1] for cell in hull_cells]
        y0 = min(hull_y)
        x0 = []
        for x in range(self.grid_size):
            if (x, y0) in hull_cells:
                x0.append(x)
        x0 = min(x0)

        return (x0, y0)

def dist(cell1, cell2, kind='eucl'):
    """
    Calculate distance of specified kind between cell1 and cell2.
    'eucl' - Euclidean distance (default)
    'taxi' - taxicab distance (Manhattan distance)
    """
    if kind == 'eucl':
        return math.sqrt( (cell1[0] - cell2[0])**2 + (cell1[1] - cell2[1])**2 )
    elif kind == 'taxi':
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])
