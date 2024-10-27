from ultralytics import YOLO
import cv2
import os

# Load the trained YOLO model
model = YOLO('D:/Projects/Voxel51/models/best.pt')  # Update to correct weights path

# Define directories
image_dir = "D:/Projects/Voxel51/new_images"  # Folder with new images
output_dir = "D:/Projects/Voxel51/detections"  # Output folder for annotated images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the input directory
for img_name in os.listdir(image_dir):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
        # Construct full path for the image
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)  # Load the image

        # Ensure the image was read properly
        if img is None:
            print(f"Warning: Could not read image {img_name}. Skipping.")
            continue

        # Perform inference
        results = model(img, conf=0.5)  # Adjust confidence threshold as needed

        # Display the detection results (optional)
        results.show()

        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, f"detection_{img_name}")
        results.save(save_dir=output_dir)  # YOLO handles saving internally
        print(f"Saved: {output_path}")
