"""
Module: image_utils

Description:
    Utility functions for image display, saving, and timing.
    
Functions:
    - show_image: Displays an image in a window for a specified duration.
    - save_image: Saves an image to the specified filepath.
    - log_elapsed_time: Logs the time taken for a specific task.
"""

import cv2
import numpy as np
import time
import os

def show_image(window_name: str, image: np.ndarray, wait_time: int = 1000):
    """
    Displays an image in a window for a specified duration.
    
    Args:
        window_name: The name of the display window.
        image: The image to be displayed.
        wait_time: Time in milliseconds to wait before closing the window. If 0, waits indefinitely.
    """
    if image is None:
        print(f"Error: Image for '{window_name}' is None.")
        return

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
        cv2.waitKey(0) 

def save_image(image: np.ndarray, filepath: str):
    """
    Saves an image to the specified filepath.
    
    Args: 
        image: The image to be saved.
        filepath: The path where the image will be saved.
    """
    
    if image is None:
        print(f"Error: Cannot save None image to {filepath}")
        return
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    cv2.imwrite(filepath, image)
    print(f"Saved image to: {filepath}")

def log_elapsed_time(start_time: float, task_name: str) -> float:
    """
    Logs the time taken for a specific task.
    Args:
        start_time: The starting time of the task.
        task_name: A descriptive name of the task.
    Returns:
        The end time after logging.
    """
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"--- Task '{task_name}' completed in {elapsed:.4f} seconds ---")
    return end_time