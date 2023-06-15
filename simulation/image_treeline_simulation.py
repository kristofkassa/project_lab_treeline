from simulation.simulation_strategy import SimulationStrategy
from PIL import Image
import numpy as np
import os
import glob

class ImageTreelineSimulationStrategy(SimulationStrategy):
    """
    Real Treeline simulation strategy.
    """

    def __init__(self):
        super().__init__()
        self.image_dir = "./simulation/real_treeline/"
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_index = 0
        self.image_path = self.image_files[self.image_index]

    def nextImage(self):
        self.image_index = (self.image_index + 1) % len(self.image_files)  # Cycle to the next image
        self.image_path = self.image_files[self.image_index]
        self.simulatePopularization()
        return self.image_path

    def simulatePopularization(self):
        self.changes = set()
        self.occupied_cells = self.preprocess_image(self.image_path, output_size=(self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.occupied_cells[i][j]:
                    self.changes.add((i, j))

    def preprocess_image(self, image_path, output_size=(150, 150), green_threshold=80):
        image = Image.open(image_path)

        # Rotate the image 90 degrees counterclockwise
        image = image.rotate(90, expand=True)
        
        # Mirror the image along the x-axis
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image = image.resize(output_size)
        image_data = np.asarray(image, dtype=np.uint8)
        
        # Extract the green channel
        green_channel = image_data[:, :, 1]
        
        # Create a boolean matrix based on the green channel
        occupied_cells = (green_channel > green_threshold) == False
        return occupied_cells