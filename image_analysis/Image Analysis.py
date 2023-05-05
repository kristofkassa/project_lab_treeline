!pip install Pillow
from PIL import Image

def create_pixel_image(image_path, pixel_size, threshold):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Compute the number of pixels in the output image
    pixel_width = (width + pixel_size - 1) // pixel_size
    pixel_height = (height + pixel_size - 1) // pixel_size
    
    # Create a new image with the computed size and white background
    pixel_image = Image.new('RGB', (pixel_width, pixel_height), 'white')
    
    # Iterate over the pixels in the output image
    for y in range(pixel_height):
        for x in range(pixel_width):
            # Compute the average color of the corresponding block of pixels
            block_color = (0, 0, 0)
            pixel_count = 0
            for i in range(pixel_size):
                for j in range(pixel_size):
                    px = x*pixel_size+i
                    py = y*pixel_size+j
                    if (px < width) and (py < height):
                        block_color = tuple(map(sum, zip(block_color, image.getpixel((px, py)))))
                        pixel_count += 1
            # Use the average color of the block to determine the pixel color
            pixel_color = tuple(c//pixel_count for c in block_color)
            # Check if the pixel is dark green (forest)
            if pixel_color[1] >= threshold and pixel_color[0] < pixel_color[2]:
                pixel_color = (0, 255, 0)  # Set green color for forest
            else:
                pixel_color = (128, 128, 128)  # Set grey color for plane
            
            # Set the pixel color in the output image
            pixel_image.putpixel((x, y), pixel_color)
    
    return pixel_image

# Load the original image
map_image = "C:/Users/HP/Downloads/mapimage1.jpg"

# Define the pixel size for the pixel art image
pixel_size = 5

# Define the threshold value for dark green color
threshold = 50

# Create the pixel art image
pixel_image = create_pixel_image(map_image, pixel_size, threshold)

# Save the pixel art image
pixel_image.save("C:/Users/HP/Downloads/mapimage_pixel.jpeg")

# Display the pixel art image
pixel_image.show()