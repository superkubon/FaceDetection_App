import os
import numpy as np
import imageio
import scipy.ndimage
import cv2

# Define input and output directories
input_folder = "D:/Work/Thesis/Face_detection/FaceD/aligned_output"
output_folder = "D:/Work/Thesis/image-to-gcode/images"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to convert RGB to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Dodge function for sketch effect (bolder lines with controlled contrast)
def dodge(front, back, strength):
    # Apply the dodge effect with a subtle contrast enhancement
    result = front * (255 / (255 - back))
    result[result > 255] = 255
    result[back == 255] = 255
    
    # Increase contrast gently (avoid overexposure)
    result = np.clip(result * strength, 0, 255)  # Apply mild enhancement to the sketch
    return result.astype('uint8')

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read and process image
        img = imageio.imread(input_path)
        gray = rgb2gray(img)
        inverted = 255 - gray

        # Apply a sharper blur (reduce sigma for sharper lines)
        blur = scipy.ndimage.gaussian_filter(inverted, sigma=40)  # Sharper blur for bold lines

        # Apply dodge function with enhanced lines
        sketch = dodge(blur, gray, strength=1)  # Use a milder contrast enhancement

        # Save output
        cv2.imwrite(output_path, sketch)
        print(f"Processed {filename}")
