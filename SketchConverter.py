import os
import numpy as np
import imageio
import scipy.ndimage
import cv2

# Define input and output directories
input_folder = "D:/Work/Thesis/Face_detection/FaceD/final_output"
output_folder = "D:/Work/Thesis/image-to-gcode/images"

# Optional resize dimensions (set to None to skip resizing)
resize_width = 800   # or None
resize_height = 600  # or None

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to convert RGB to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Dodge function for sketch effect
def dodge(front, back, strength):
    result = front * (255 / (255 - back))
    result[result > 255] = 255
    result[back == 255] = 255
    result = np.clip(result * strength, 0, 255)
    return result.astype('uint8')

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read and optionally resize image
        img = imageio.imread(input_path)
        if resize_width and resize_height:
            img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale and apply effects
        gray = rgb2gray(img)
        inverted = 255 - gray
        blur = scipy.ndimage.gaussian_filter(inverted, sigma=40)
        sketch = dodge(blur, gray, strength=1)

        # Save result
        cv2.imwrite(output_path, sketch)
        print(f"Processed {filename}")
