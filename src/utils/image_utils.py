# src/utils/image_utils.py

import cv2
import numpy as np
import time
import os

def show_image(window_name, image, wait_time=1000):
    """Displays an image in a window."""
    if image is None:
        print(f"Error: Image for '{window_name}' is None.")
        return
    # Resize for better display if image is too large
    max_display_height = 800
    max_display_width = 1200
    height, width = image.shape[:2]

    if height > max_display_height or width > max_display_width:
        scale_w = max_display_width / width
        scale_h = max_display_height / height
        scale = min(scale_w, scale_h)
        dim = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    cv2.imshow(window_name, resized_image)
    if wait_time > 0:
        cv2.waitKey(wait_time)
    else:
        cv2.waitKey(0) # Wait indefinitely until a key is pressed

def save_image(image, filepath):
    """Saves an image to the specified filepath."""
    if image is None:
        print(f"Error: Cannot save None image to {filepath}")
        return
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    cv2.imwrite(filepath, image)
    print(f"Saved image to: {filepath}")

def log_elapsed_time(start_time, task_name):
    """Logs the time taken for a specific task."""
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"--- Task '{task_name}' completed in {elapsed:.4f} seconds ---")
    return end_time # Return end_time to be used as start_time for the next task