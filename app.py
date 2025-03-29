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

# Step 3: Define the Face Alignment Function
def align_face(image, face):
    """
    Aligns the face based on detected landmarks, specifically the eyes.
    This assumes that the 'face' object contains the detected keypoints (landmarks).
    """
    # Get the coordinates of the eyes from the face landmarks
    left_eye = tuple(face.kps[0].astype(int))  # Left eye (first point)
    right_eye = tuple(face.kps[1].astype(int))  # Right eye (second point)

    # Compute the center of the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)  # Use normal division

    # Convert to integers
    eyes_center = tuple(map(int, eyes_center))  # Ensure the center is a tuple of integers

    # Compute the angle between the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the rotation matrix for the face alignment
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)

    # Get the aligned face by rotating the image around the eyes center
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_face


# Step 4: Traverse through each image in the input directory
for image_filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_filename)
    
    # Check if the file is an image (optional filter, e.g., only .jpg, .png)
    if image_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        print(f"Processing image: {image_path}")
        
        # Step 5: Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Image {image_filename} not found!")
            continue
        else:
            print(f"Image loaded successfully! Image shape: {image.shape}")

        # Step 6: Use rembg for Background Removal
        print("Starting background removal using rembg...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        pil_image = Image.fromarray(image_rgb)  # Convert to PIL image

        # Use rembg to remove the background
        output_image = rembg.remove(pil_image)

        # Convert the output image to bytes using BytesIO
        output_bytes_io = io.BytesIO()
        output_image.save(output_bytes_io, format="PNG")
        output_bytes = output_bytes_io.getvalue()

        # Step 7: Save the Output of Background Removal to a Temporary File
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(output_bytes)
            temp_file_path = temp_file.name
        
        # Open the temporary file with PIL
        output_image = Image.open(temp_file_path)

        # Step 8: Perform Face Detection
        print("Starting face detection...")
        output_image_rgb = output_image.convert("RGB")  # Convert to RGB (removes alpha channel)
        faces = face_analyzer.get(np.array(output_image_rgb))

        # Debugging: Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected in the image.")
        else:
            print(f"Detected {len(faces)} faces.")

        # Step 9: Draw Bounding Boxes and Landmarks on the Image
        image_with_faces = np.array(output_image_rgb)  # Convert PIL image to numpy array for OpenCV
        for face in faces:
            # At this point, no bounding box is drawn during detection.
            # Instead, we will align the face and draw the bounding box after alignment.

            # Align face using the detected landmarks (we'll align to the eyes)
            aligned_face = align_face(image_with_faces, face)

            # Step 10: Draw the bounding box after alignment
            new_bbox = face.bbox.astype(int)

            # Option 1: Scale the bounding box by a factor (e.g., 1.2 for 20% larger)
            scale_factor = 1.75  # Change this value as needed (1.0 means no change)
            width = new_bbox[2] - new_bbox[0]
            height = new_bbox[3] - new_bbox[1]

            # Calculate the new width and height based on the scale factor
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Calculate the new top-left and bottom-right corners to maintain the center
            center_x = (new_bbox[0] + new_bbox[2]) // 2
            center_y = (new_bbox[1] + new_bbox[3]) // 2

            # Calculate the new bounding box coordinates
            new_x1 = center_x - new_width // 2
            new_y1 = center_y - new_height // 2
            new_x2 = center_x + new_width // 2
            new_y2 = center_y + new_height // 2

            # Option 2: Add padding to the bounding box (e.g., 10 pixels)
            padding = 10  # Change this value to control the padding
            new_x1 = max(new_bbox[0] - padding, 0)
            new_y1 = max(new_bbox[1] - padding, 0)
            new_x2 = new_bbox[2] + padding
            new_y2 = new_bbox[3] + padding

            # Ensure the new bounding box is within image bounds
            new_x2 = min(new_x2, image.shape[1])
            new_y2 = min(new_y2, image.shape[0])

            # Draw the modified bounding box
            cv2.rectangle(aligned_face, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

            # Step 11: Save the Output Images
            # Save the background-removed image (PNG)
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_bg_removed.png")
            output_image.save(output_image_path)
            
            # Save the image with detected faces and landmarks (JPEG)
            output_detected_faces_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_faces_detected.jpg")
            # Convert to RGB before saving as JPEG (OpenCV uses BGR)
            image_with_faces_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_detected_faces_path, image_with_faces_rgb)

            # Save the aligned face image (JPEG)
            output_aligned_face_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_aligned_face.jpg")
            cv2.imwrite(output_aligned_face_path, aligned_face)

            print(f"Processed images saved:\n - Background removed: {output_image_path}\n - Detected faces: {output_detected_faces_path}\n - Aligned face: {output_aligned_face_path}")

print("All images processed.")
