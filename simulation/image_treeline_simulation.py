from simulation.simulation_strategy import SimulationStrategy
from PIL import Image
import numpy as np

class ImageTreelineSimulationStrategy(SimulationStrategy):
    """
    Real Treeline simulation strategy.
    """

    def __init__(self):
        super().__init__()
        self.image_path = "./simulation/forest.png"

    def simulatePopularization(self):
        self.changes = set()
        self.occupied_cells = self.preprocess_image(self.image_path, output_size=(self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i][j]:
                    self.changes.add((i, j))


    def preprocess_image(self, image_path, output_size=(150, 150), green_threshold=80):
        image = Image.open(image_path)
        image = image.resize(output_size)
        image_data = np.asarray(image, dtype=np.uint8)
        
        # Extract the green channel
        green_channel = image_data[:, :, 1]
        
        # Create a boolean matrix based on the green channel
        occupied_cells = (green_channel > green_threshold) == False
        return occupied_cells