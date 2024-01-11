from abc import abstractmethod
from datetime import datetime
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd

class SimulationStrategy:

    def __init__(self):
        sys.setrecursionlimit(9*10**8)
        self.grid_size = 2**8
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.population_data = []
        self.changes = set()
        self.e = 0.2
        self.c = 0.8
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.simple_cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull_list = []

        self.simple_hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.simple_hull_list = []

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

    def _get_neighbors(self, i, j, neighborhood = "simple"):
        n = self.grid_size
        neighbors = []

        directions = [
            ((i-1), j),
            (i, (j+1)%n),
            ((i+1), j),
            (i, (j-1)%n)
        ] if neighborhood != "simple" else [
            ((i-1), j),
            (i, (j+1)),
            ((i+1), j),
            (i, (j-1))
        ]

        for x, y in directions:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                neighbors.append((x, y))
        return neighbors

    def _dfs(self, i, j, visited, cluster, neighborhood = "simple"):
        visited.add((i, j))
        cluster.add((i, j))
        for x, y in self._get_neighbors(i, j, neighborhood):
            if self.occupied_cells[x, y] and (x, y) not in visited:
                self._dfs(x, y, visited, cluster, neighborhood)

    def identifyPercolationClusters(self):
        self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.simple_cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        #Cylindrical giant cluster
        visited = set()
        clusters = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i, j] and (i, j) not in visited:
                    cluster = set()                    
                    self._dfs(i, j, visited, cluster, "cylindrical")
                    clusters.append(cluster)

        if not clusters:
            self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        else:
            largest_cluster = max(clusters, key=len)
            self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for i, j in largest_cluster:
                self.cluster[i, j] = True
        
        #print(self.cluster)

        #Simple giant cluster
        visited = set()
        clusters = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i, j] and (i, j) not in visited:
                    cluster = set()
                    self._dfs(i, j, visited, cluster, "simple")
                    clusters.append(cluster)

        if not clusters:
            self.simple_cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        else:
            largest_cluster = max(clusters, key=len)
            self.simple_cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for i, j in largest_cluster:
                self.simple_cluster[i, j] = True
        
        #print(self.simple_cluster)
        #print(np.array_equal(self.cluster, self.simple_cluster))

    def _next_edge_node(self, i, j, prev_i, prev_j, neighborhood = "simple"):
        n = self.grid_size
        directions = [
            ((i-1), j),
            (i, (j+1)%n),
            ((i+1), j),
            (i, (j-1)%n)
        ] if neighborhood != "simple" else [
            ((i-1), j),
            (i, (j+1)),
            ((i+1), j),
            (i, (j-1))
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
        self.simple_hull = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.hull_list = []
        self.simple_hull_list = []

        # Find an initial edge node arbitrarily for cylindrical topology
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                if self.cluster[i, j]:
                    start_i, start_j = i, j
                    break
            else:
                continue
            break

        # Find the last edge node arbitrarily for cylindrical topology
        for j in reversed(range(self.grid_size)):
            for i in range(self.grid_size):
                if self.cluster[i, j]:
                    end_i, end_j = i, j
                    break
            else:
                continue
            break

       # print(start_i, start_j, end_i, end_j)

        # Start edge following for cylindrical case
        prev_i, prev_j = start_i, start_j
        curr_i, curr_j = self._next_edge_node(start_i, start_j, start_i, start_j, "cylindrical")
        self.hull[start_i, start_j] = True
        self.hull_list.append((start_i, start_j))

        while self.hull[end_i, end_j] == 0 or (curr_i != start_i or curr_j != start_j):
            self.hull[curr_i, curr_j] = True
            self.hull_list.append((curr_i, curr_j))
            next_i, next_j = self._next_edge_node(curr_i, curr_j, prev_i, prev_j, "cylindrical")
            prev_i, prev_j = curr_i, curr_j
            curr_i, curr_j = next_i, next_j

        self.hull[curr_i, curr_j] = True  # Mark the starting node again to close the hull
        self.hull_list.append((curr_i, curr_j))

        #print(self.hull_list)

        # Find an initial edge node arbitrarily for simple topology
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                if self.simple_cluster[i, j]:
                    start_i, start_j = i, j
                    break
            else:
                continue
            break

        # Find the last edge node arbitrarily for simple topology
        for j in reversed(range(self.grid_size)):
            for i in range(self.grid_size):
                if self.simple_cluster[i, j]:
                    end_i, end_j = i, j
                    break
            else:
                continue
            break

        #Start edge following for simple case 
        prev_i, prev_j = start_i, start_j
        curr_i, curr_j = self._next_edge_node(start_i, start_j, start_i, start_j, "simple")
        self.simple_hull[start_i, start_j] = True
        self.simple_hull_list.append((start_i, start_j))

        while self.simple_hull[end_i, end_j] == 0: #or (curr_i != start_i or curr_j != start_j):
            self.simple_hull[curr_i, curr_j] = True
            self.simple_hull_list.append((curr_i, curr_j))
            next_i, next_j = self._next_edge_node(curr_i, curr_j, prev_i, prev_j, "simple")
            prev_i, prev_j = curr_i, curr_j
            curr_i, curr_j = next_i, next_j

        self.simple_hull[curr_i, curr_j] = True  # Mark the starting node again to close the hull
        self.simple_hull_list.append((curr_i, curr_j))

        #print(self.simple_hull_list)


    def calculate_fractal_dimensions(self):
        self.markHull()
        return self.calculate_fractal_dimension_boxcounting(), self.calculate_fractal_dimension_correlation(), self.calculate_fractal_dimension_ruler(), self.calculate_fractal_dimension_avgdist()

    def calculate_fractal_dimension_boxcounting(self):
        """
            Initialize the box sizes and counts.
            Iterate through the different box sizes.
            For each box size, cover the hull with non-overlapping boxes.
            Count the number of boxes that intersect with the hull.
            Calculate the slope of the log-log plot of box size versus the count of boxes that intersect the hull.
        """

        log_box_sizes, log_box_counts = self.get_box_details()

        #plot the data points
        plt.figure(figsize=(8, 6))
        plt.plot(log_box_sizes[2:-4], log_box_counts[2:-4], 'o', color = 'lime')
        plt.plot([log_box_sizes[i] for i in [0,1,-4,-3,-2,-1]], [log_box_counts[i] for i in [0,1,-4,-3,-2,-1]], 'o', color = 'gray')
        plt.title('Log-Log Plot of Box Sizes vs. Box Counts')
        plt.xlabel('Log(Box Sizes)')
        plt.ylabel('Log(Box Counts)')

        # Calculate the slope of the log-log plot using linear regression
        slope, intercept = np.polyfit(log_box_sizes[2:-4], log_box_counts[2:-4], 1)
        fractal_dimension = -slope
        # print("Box Dimension:", fractal_dimension)


        #plot the actual and desired regression line
        regression_line = slope * np.array(log_box_sizes) + intercept
        magic_line = (-1.75) * np.array(log_box_sizes) + intercept
        plt.plot(log_box_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_box_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()

        return fractal_dimension
    
    def get_box_details(self):
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

        log_box_sizes = [math.log2(size) for size in box_sizes]
        log_box_counts = [math.log2(count) for count in box_counts]

        return log_box_sizes, log_box_counts
    
    def calculate_fractal_dimension_correlation(self):
        """
        Calculate the fractal dimension using the correlation dimension method.
        This method computes the correlation sum C(r) which counts the number of point pairs that have distance less than r.
        Then it estimates the dimension as the slope of log(C(r)) vs log(r).
        """

        log_radius_sizes, log_correlation_sums = self.get_correlation_details()

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
        # print("Correlation Dimenson:", fractal_dimension)


        #plot the actual and desired regression line
        regression_line = slope * np.array(log_radius_sizes) + intercept
        magic_line = (1.75) * np.array(log_radius_sizes) + intercept
        plt.plot(log_radius_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_radius_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()

        return fractal_dimension
    
    def get_correlation_details(self):
        
        max_radius = self.grid_size // 2
        radius_sizes = [2 ** i for i in range(0,int(math.log2(max_radius)))]
        correlation_sums = []

        hull_cells = list(set(self.hull_list)) #unique list of hull cells
        hull_num = len(hull_cells)
        distances = np.zeros((hull_num, hull_num), dtype=float)
        # print("Distance matrix calculation has started.")
        for i in range(hull_num):
            # print(f"i = {i}")
            x1, y1 = hull_cells[i]
            for j in range(i+1, hull_num):
                x2, y2 = hull_cells[j]
                distances[i,j] = distances[j,i] = (x2 - x1)**2 + (y2 - y1)**2

        for radius in radius_sizes:
            # print(f"CorrSum({radius}) is being counted.")
            x = (np.count_nonzero(distances <= radius**2) - hull_num)/2            
            correlation_sums.append(x)

        log_radius_sizes = [math.log2(size) for size in radius_sizes]
        log_correlation_sums = [math.log2(correlation_sum) for correlation_sum in correlation_sums]

        return log_radius_sizes, log_correlation_sums

    def calculate_fractal_dimension_ruler(self):

        log_ruler_sizes, log_ruler_counts = self.get_ruler_details()

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
        # print("Ruler Dimension:", fractal_dimension)


        regression_line = slope * np.array(log_ruler_sizes) + intercept
        magic_line = (-1.75) * np.array(log_ruler_sizes) + intercept
        plt.plot(log_ruler_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {slope:.2f})')
        plt.plot(log_ruler_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')

        plt.legend()
        plt.grid(True)
        plt.show()
        return fractal_dimension
       
    def get_ruler_details(self):
        
        max_size = self.grid_size #//4
        ruler_sizes = [2 ** i for i in range(0, int(math.log2(max_size)) )]
        ruler_counts = []

        hull_cells = self.simple_hull_list

        for ruler_size in ruler_sizes:
            _ = 0
            ruler_count = 0
            # print("r=", ruler_size)
            cell0 = hull_cells[0]
            i = 1
            while i < len(hull_cells):
                while (i < len(hull_cells)) and (dist(cell0, hull_cells[i]) <= ruler_size):
                    i+=1
                # print(cell0)
                cell0 = hull_cells[i-1]
                ruler_count += 1

                _ += 1
                if _ > self.grid_size ** 2: 
                    # print("INFINITE LOOP")
                    break #to avoid inifite loops in developement stage
            #print(hull_cells[i-1])
            ruler_counts.append(ruler_count)
            # print("count = ", ruler_count, "\n")

        log_ruler_sizes = [math.log2(ruler_size) for ruler_size in ruler_sizes]
        log_ruler_counts = [math.log2(ruler_count) for ruler_count in ruler_counts]

        return log_ruler_sizes, log_ruler_counts

    def calculate_fractal_dimension_avgdist(self):

        log_k_lengths, log_avg_dists = self.get_avg_dist_details()

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
        # print("AvgDist Dimension:", fractal_dimension)


        regression_line = slope * np.array(log_k_lengths) + intercept
        magic_line = (1/1.75) * np.array(log_k_lengths) + intercept
        plt.plot(log_k_lengths, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = 1/{fractal_dimension:.2f})')
        plt.plot(log_k_lengths, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = 1/1.75)')

        plt.legend()
        plt.grid(True)
        plt.show()
        return fractal_dimension
    
    def get_avg_dist_details(self):
        
        k_lengths = [2 ** i for i in range(0, int(math.log2(len(self.simple_hull_list)//2)) )]
        avg_dists = []

        for k in k_lengths:
            m = len(self.simple_hull_list) // k
            avg = 0.0
            for i in range(1, m):
                # print("k=",k,", i*k=",i*k,", m=",m)
                avg += dist(self.simple_hull_list[i*k], self.simple_hull_list[(i-1)*k])
            avg /= m
            # print("k=", k, "  avg=", avg)

            avg_dists.append(avg)

        log_k_lengths = [math.log2(k) for k in k_lengths]
        log_avg_dists = [math.log2(avg) for avg in avg_dists]

        return log_k_lengths, log_avg_dists

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

    def autoSimulate(self):

        box_results = []
        corr_results = []
        ruler_results = []
        avgdist_results = []

        sample_size = 100

        print(f"Mass simulation has started with grid_size = 2^{int(math.log2(self.grid_size))}, sample size = {sample_size}.")
        for counter in range(sample_size):
            print(f"Iteration no. {counter+1} has begun.")
            self.cluster = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            self.occupied_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            self.simulatePopularization()
            self.identifyPercolationClusters()
            self.markHull()

            box = self.get_box_details()
            box_results.append(box)

            corr = self.get_correlation_details()
            corr_results.append(corr)

            ruler = self.get_ruler_details()
            ruler_results.append(ruler)

            avgdist = self.get_avg_dist_details()
            avgdist_results.append(avgdist)

        self.create_excel_file(box_results, corr_results, ruler_results, avgdist_results)
    
    def create_excel_file(self, box_results, corr_results, ruler_results, avgdist_results):

        folder_path = os.path.join(os.getcwd(), 'simulation_results')

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"simulation_results_{timestamp}.xlsx"
        file_path = os.path.join(folder_path, file_name)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Convert each list of results to a DataFrame and save to Excel
            pd.DataFrame(box_results).to_excel(writer, sheet_name='box_counting')
            pd.DataFrame(corr_results).to_excel(writer, sheet_name='correlation')
            pd.DataFrame(ruler_results).to_excel(writer, sheet_name='ruler')
            pd.DataFrame(avgdist_results).to_excel(writer, sheet_name='avgdist')

        print(f"Excel file created at {file_path}")
    
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