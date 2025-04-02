import insightface
import cv2
import numpy as np
import rembg
from PIL import Image
import io
import os

# Step 1: Initialize Face Detection Model (RetinaFace)
print("Initializing face analysis...")
face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')

# Try GPU first, fallback to CPU if needed
try:
    face_analyzer.prepare(ctx_id=0)  # GPU
except Exception as e:
    print("GPU initialization failed. Switching to CPU...")
    face_analyzer.prepare(ctx_id=-1)  # CPU fallback

# Define Directories
input_dir = "D:\\Work\\Thesis\\Face_detection\\FaceD\\input\\"
cam_input_dir = "D:\\Work\\Thesis\\Face_detection\\FaceD\\cam_input\\"
output_dir = "D:\\Work\\Thesis\\Face_detection\\FaceD\\output\\"

# Ensure output directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(cam_input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Step 2: Define the Face Alignment Function
def align_face(image, face):
    left_eye = tuple(face.kps[0].astype(int))
    right_eye = tuple(face.kps[1].astype(int))
    eyes_center = tuple(map(int, ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)))
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)

    # Convert to float for more precision during rotation
    image_float = image.astype(np.float32)

    # Use INTER_CUBIC for better quality rotation
    aligned_face = cv2.warpAffine(image_float, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    # Convert back to uint8 (standard image format)
    aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)

    return aligned_face

# Step 3: Process Images (Folder or Webcam)
def process_images(mode=1):
    if mode == 1:  # Process images from input folder
        for image_filename in os.listdir(input_dir):
            if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(input_dir, image_filename)
            process_image(image_path, image_filename)
    elif mode == 2:  # Capture images from webcam and save them to cam_input folder
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        print("Press 'q' to stop capturing images.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Save the captured frame to the cam_input folder
            timestamp = int(cv2.getTickCount())  # Use timestamp for unique filename
            cam_image_path = os.path.join(cam_input_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(cam_image_path, frame)
            print(f"Captured image: {cam_image_path}")

            # Process the captured image
            process_image(cam_image_path, f"frame_{timestamp}.jpg")

            # Display the webcam feed (optional)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
                break

        cap.release()
        cv2.destroyAllWindows()

# Step 4: Process each image
def process_image(image_path, image_filename):
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_filename}")
        return

    print(f"Image loaded successfully: {image.shape}")
    
    # Step 5: Background Removal
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with PIL
    pil_image = Image.fromarray(image_rgb)
    print("Removing background...")
    output_image = rembg.remove(pil_image)
    
    # Use in-memory buffer instead of temp file
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    output_image = Image.open(io.BytesIO(output_buffer.getvalue())).convert("RGB")

    # Save background-removed image
    bg_removed_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_bg_removed.png")
    output_image.save(bg_removed_path)
    print(f"Background removed image saved: {bg_removed_path}")

    # Step 6: Face Detection with Bounding Box
    output_np = np.array(output_image)
    print("Detecting faces...")
    faces = face_analyzer.get(output_np)

    if not faces:
        print("No faces detected.")
        return
    print(f"Detected {len(faces)} face(s).")

    # Save face detection output image (just drawing bounding boxes)
    face_detected_image = output_np.copy()
    for face in faces:
        bbox = face.bbox.astype(int)
        # Draw bounding box around detected face
        cv2.rectangle(face_detected_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Save the image with bounding boxes
    face_detected_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_faces_detected.jpg")
    cv2.imwrite(face_detected_path, cv2.cvtColor(face_detected_image, cv2.COLOR_RGB2BGR))
    print(f"Face detection with bounding box saved: {face_detected_path}")

    # Step 7: Face Alignment without Bounding Box
    for i, face in enumerate(faces):
        aligned_face = align_face(output_np, face)

        # Ensure color consistency by converting back to RGB
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Save aligned face without bounding box
        aligned_face_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_aligned_face_{i}.jpg")
        cv2.imwrite(aligned_face_path, cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR))
        print(f"Aligned face {i} saved: {aligned_face_path}")

print("✅ All images processed.")

# Step 8: Choose Mode (1 or 2)
mode = int(input("Enter mode (1 for folder processing, 2 for webcam processing): "))
process_images(mode)
