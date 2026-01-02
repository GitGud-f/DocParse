# main.py

import cv2
import os
import numpy as np
import time

from src.scanner.geometry import detect_document_corners, four_point_transform
from src.scanner.filters import adaptive_thresholding, clahe_equalization, preprocess_image
from src.utils.image_utils import show_image, save_image, log_elapsed_time

def process_document_phase1(image_path, output_dir="data/output/phase1_processed"):
    """
    Executes Phase I: Geometric & Illumination Preprocessing for a single image.
    """
    print(f"\n--- Starting Phase I for: {image_path} ---")
    start_time = time.time()

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Display original image
    show_image("Original Image", original_image, wait_time=0)
    
    # --- Geometric Correction ---
    print("Step 1: Performing Geometric Correction...")
    geo_start_time = time.time()
    
    # Convert to grayscale for corner detection
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Find corners
    corners = detect_document_corners(original_image, debug=True)
    
    if corners is None:
        print("Could not detect document corners. Skipping perspective transform.")
        warped_image = original_image # Use original if corners not found
        # You might want to save the original image in the processed dir as a fallback
        save_image(warped_image, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_fallback.jpg"))
        geo_end_time = log_elapsed_time(geo_start_time, "Geometric Correction (Fallback)")
        
        # Proceed to illumination with original image if no corners
        processed_image = warped_image
    else:
        # Draw corners for visualization
        corner_image = original_image.copy()
        for pt in corners:
            cv2.circle(corner_image, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        show_image("Detected Corners", corner_image, wait_time=0)
        save_image(corner_image, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_corners.jpg"))

        # Perform perspective transform
        warped_image = four_point_transform(original_image, corners)
        show_image("Warped Image (Geometric Correction)", warped_image, wait_time=0)
        save_image(warped_image, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_warped.jpg"))
        geo_end_time = log_elapsed_time(geo_start_time, "Geometric Correction")
    
    # --- Illumination Normalization ---
    print("Step 2: Performing Illumination Normalization...")
    illum_start_time = time.time()
    
    # Apply CLAHE for contrast enhancement
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe_equalization(warped_gray)
    show_image("CLAHE Enhanced Grayscale", enhanced_gray, wait_time=0)
    save_image(enhanced_gray, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_clahe.jpg"))

    # Apply adaptive thresholding for binarization (often good for OCR)
    # Adjust block_size and c based on image characteristics
    binarized_image = adaptive_thresholding(enhanced_gray, block_size=15, c=4)
    show_image("Binarized Image (Adaptive Thresholding)", binarized_image, wait_time=0)
    save_image(binarized_image, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_binarized.jpg"))

    illum_end_time = log_elapsed_time(illum_start_time, "Illumination Normalization")
    
    # For Phase I, the best output to save might be the binarized image,
    # as it's commonly used for OCR. However, sometimes the enhanced grayscale
    # image is also useful. Let's save the binarized one as the "processed" output.
    processed_image = binarized_image # Or enhanced_gray if preferred

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    final_processed_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_phase1_processed.jpg")
    save_image(processed_image, final_processed_path)

    final_end_time = log_elapsed_time(start_time, f"Phase I for {image_path}")
    
    print(f"--- Phase I completed. Processed image saved to: {final_processed_path} ---")
    
    # Return the processed image for the next phase (though in this structure, we'd likely save and reload)
    return processed_image


if __name__ == "__main__":
    # --- Configuration ---
    RAW_IMAGES_DIR = "data/raw"
    PHASE1_OUTPUT_DIR = "data/processed/phase1" # Renamed from 'output' for clarity
    
    # Create dummy raw data if it doesn't exist
    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    if not os.listdir(RAW_IMAGES_DIR):
        print("No raw images found. Creating a dummy image for demonstration.")
        # Create a simple white canvas
        dummy_img = np.full((1000, 800, 3), 255, dtype=np.uint8)
        # Draw a tilted rectangle to simulate a document
        pts = np.array([[100, 50], [700, 150], [650, 900], [50, 800]], dtype=np.int32)
        cv2.fillPoly(dummy_img, [pts], (200, 200, 200)) # Gray document
        cv2.putText(dummy_img, "Sample Document", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
        dummy_image_path = os.path.join(RAW_IMAGES_DIR, "dummy_document.jpg")
        cv2.imwrite(dummy_image_path, dummy_img)
        print(f"Dummy image created at: {dummy_image_path}")
        
    # --- Process images ---
    images_to_process = [os.path.join(RAW_IMAGES_DIR, f) for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images_to_process:
        print(f"No image files found in {RAW_IMAGES_DIR}. Please add some images.")
    else:
        for img_path in images_to_process:
            processed_img = process_document_phase1(img_path, PHASE1_OUTPUT_DIR)
            if processed_img is not None:
                # Here, `processed_img` is the output of phase 1.
                # For the next phase, you'd typically save this image and then load it.
                # The `save_image` call inside `process_document_phase1` does this.
                pass
    
    print("\n--- Phase I Execution Finished ---")
    print("Check 'data/processed/phase1/' for the results.")
    cv2.destroyAllWindows() # Close all OpenCV windows at the end