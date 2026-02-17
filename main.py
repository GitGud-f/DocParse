# main.py

import cv2
import os
import numpy as np
import time

from src.scanner.geometry import detect_document_corners, four_point_transform
from src.scanner.filters import adaptive_thresholding, clahe_equalization, preprocess_image
from src.utils.image_utils import show_image, save_image, log_elapsed_time
from src.utils.config import cfg 


def process_document_phase1(image_path, output_dir=None):
    """
    Executes Phase I: Geometric & Illumination Preprocessing for a single image.
    """
    if output_dir is None: output_dir = cfg['paths']['processed_data']
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
    try:
        corners = detect_document_corners(original_image)
    except Exception as e:
        print(f"Warning: Detection crashed with error {e}. Proceeding with fallback.")
        corners = None
    
    if corners is not None:
        print("Corners detected. Applying perspective transform.")
        # Draw corners for debug saving
        corner_image = original_image.copy()
        cv2.drawContours(corner_image, [corners.astype(int)], -1, (0, 255, 0), 3)
        save_image(corner_image, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_corners.jpg"))

        # Apply Transform
        warped_image = four_point_transform(original_image, corners)
    else:
        # --- FALLBACK HANDLING ---
        print("⚠️ Warning: Document corners not found.") 
        print("-> Fallback: Using original image as 'warped' image.")
        print("-> Assumption: User has manually cropped the image or background is minimal.")
        
        # We use the original image, but we might want to resize it if it's massive 
        # (optional, depending on your OCR speed requirements)
        warped_image = original_image.copy()

    geo_end_time = log_elapsed_time(geo_start_time, "Geometric Correction (Or Fallback)")
       
    # --- Illumination Normalization ---
    print("Step 2: Performing Illumination Normalization...")
    illum_start_time = time.time()
    
    clahe_clip = cfg['illumination']['clahe']['clip_limit']
    clahe_grid = tuple(cfg['illumination']['clahe']['tile_grid_size'])
    thresh_block = cfg['illumination']['adaptive_threshold']['block_size']
    thresh_c = cfg['illumination']['adaptive_threshold']['c']
    
    # Apply CLAHE for contrast enhancement
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe_equalization(warped_gray, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
    show_image("Enhanced Grayscale (for DL)", enhanced_gray, wait_time=0)
    save_image(enhanced_gray, os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_enhanced_gray.jpg"))

    # Apply adaptive thresholding for binarization (often good for OCR)
    # Adjust block_size and c based on image characteristics
    binarized_image = adaptive_thresholding(enhanced_gray, block_size=thresh_block, c=thresh_c)
    show_image("Binarized Image (for OCR)", binarized_image, wait_time=0)
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
    RAW_IMAGES_DIR = cfg['paths']['raw_data']
    PHASE1_OUTPUT_DIR = cfg['paths']['processed_data']
    
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