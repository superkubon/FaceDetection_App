import insightface
import cv2
import numpy as np
import rembg
from PIL import Image
import io
import tempfile
import os

# Step 1: Initialize Face Detection Model (RetinaFace)
print("Initializing face analysis...")
face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0)  # Use 0 for GPU, -1 for CPU

# Step 2: Define Directories for Input and Output
input_dir = "D:\\Work\\Thesis\\Face_detection\\FaceD\\input\\"  # Change this to your input folder
output_dir = "D:\\Work\\Thesis\\Face_detection\\FaceD\\output\\"  # Change this to your output folder

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 3: Traverse through each image in the input directory
for image_filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_filename)
    
    # Check if the file is an image (optional filter, e.g., only .jpg, .png)
    if image_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        print(f"Processing image: {image_path}")
        
        # Step 4: Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Image {image_filename} not found!")
            continue
        else:
            print(f"Image loaded successfully! Image shape: {image.shape}")

        # Step 5: Use rembg for Background Removal
        print("Starting background removal using rembg...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        pil_image = Image.fromarray(image_rgb)  # Convert to PIL image

        # Use rembg to remove the background
        output_image = rembg.remove(pil_image)

        # Convert the output image to bytes using BytesIO
        output_bytes_io = io.BytesIO()
        output_image.save(output_bytes_io, format="PNG")
        output_bytes = output_bytes_io.getvalue()

        # Step 6: Save the Output of Background Removal to a Temporary File
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(output_bytes)
            temp_file_path = temp_file.name
        
        # Open the temporary file with PIL
        output_image = Image.open(temp_file_path)

        # Step 7: Perform Face Detection
        print("Starting face detection...")
        output_image_rgb = output_image.convert("RGB")  # Convert to RGB (removes alpha channel)
        faces = face_analyzer.get(np.array(output_image_rgb))

        # Debugging: Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected in the image.")
        else:
            print(f"Detected {len(faces)} faces.")

        # Step 8: Draw Bounding Boxes and Landmarks on the Image
        image_with_faces = np.array(output_image_rgb)  # Convert PIL image to numpy array for OpenCV
        for face in faces:
            # Draw the bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(image_with_faces, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Draw landmarks (eyes, nose, mouth corners)
            for point in face.kps:  # facial landmarks
                cv2.circle(image_with_faces, tuple(point.astype(int)), 2, (0, 0, 255), -1)

        # Step 9: Save the Output Images
        # Save the background-removed image (PNG)
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_bg_removed.jpg")
        output_image.save(output_image_path)
        
        # Save the image with detected faces and landmarks (JPEG)
        output_detected_faces_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_faces_detected.jpg")
        # Convert to RGB before saving as JPEG (OpenCV uses BGR)
        image_with_faces_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_detected_faces_path, image_with_faces_rgb)

        print(f"Processed images saved:\n - Background removed: {output_image_path}\n - Detected faces: {output_detected_faces_path}")

print("All images processed.")