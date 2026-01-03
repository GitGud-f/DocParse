"""
Module: hough

Description:
    Hough Transform-based functions for document corner detection.
    Includes line merging and intersection computation.
    
Functions:
    - detect_hough_corners: Detects document corners using Hough Line Transform.
    - compute_intersection: Computes intersection point of two lines in (rho, theta) format.
    - merge_related_lines: Merges similar lines based on proximity in (rho, theta) space.
    - get_processing_mask: Generates a binary mask for edge detection.
"""
import cv2
import numpy as np
from src.scanner.geometry import order_points
from src.utils.config import cfg

def compute_intersection(line1: list, line2: list) -> list:
    """
    Computes the intersection point of two lines given in (rho, theta) format.
    
    Args:
        line1: A list [rho1, theta1] representing the first line.
        line2: A list [rho2, theta2] representing the second line.
        
    Returns:
        A list [x, y] representing the intersection point, or None if lines are parallel
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    det = np.linalg.det(A)
    if abs(det) < 0.001: return None 

    x, y = np.linalg.solve(A, b)
    return [float(x), float(y)]

def merge_related_lines(lines: list) -> list:
    """
    Merges similar lines based on proximity in (rho, theta) space.
    
    Args:
        lines: A list of lines in (rho, theta) format.
        
    Returns:
        A list of merged lines.
    """
    if lines is None: return []
    
    c_hough = cfg['geometry']['hough_method']['clustering']

    normalized_lines = []
    for line in lines:
        rho, theta = line[0]
        if rho < 0:
            rho = -rho
            theta = theta - np.pi
        normalized_lines.append([rho, theta])

    # Tolerances: 20 pixels distance, 10 degrees angle
    min_rho_diff = c_hough['min_rho_diff']
    min_theta_diff = np.deg2rad(c_hough['min_theta_diff_deg'])
    clusters = []
    
    for rho, theta in normalized_lines:
        matched = False
        for cluster in clusters:
            c_rho, c_theta = np.mean(cluster, axis=0)
            if abs(rho - c_rho) < min_rho_diff and abs(theta - c_theta) < min_theta_diff:
                cluster.append([rho, theta])
                matched = True
                break
        if not matched: clusters.append([[rho, theta]])
            
    merged_lines = []
    for cluster in clusters:
        avg_line = np.mean(cluster, axis=0)
        merged_lines.append([avg_line])
    return merged_lines

def get_processing_mask(image: np.ndarray) -> np.ndarray:
    """
    Helper function to generate the binary mask used for edge detection.
    Optimized to handle both Dark (Otsu) and Light (Adaptive) backgrounds,
    with more robust morphological closing.
    
    Args:
        image: The input grayscale image.
        
    Returns:
        The binary mask image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    c_mask = cfg['geometry']['hough_method']['mask_generation']
    
    blurred = cv2.GaussianBlur(gray, tuple(c_mask['blur_ksize']), 0)
    
    # 1. (Best for Dark Backgrounds / High Contrast)
    otsu_thresh_val, otsu_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 2. Validation: Check if Otsu resulted in a mostly white or mostly black image (likely failed)
    white_pixels = np.count_nonzero(otsu_img)
    total_pixels = otsu_img.size
    white_ratio = white_pixels / total_pixels
    
    current_mask = None 

    if white_ratio > c_mask['otsu_white_ratio_upper'] or white_ratio < c_mask['otsu_white_ratio_lower']:
        # Fallback to Adaptive Thresholding (Better for Light Backgrounds / Low Contrast)
        current_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, c_mask['adaptive_block_size'], 
                                             c_mask['adaptive_c'])
    else:

        current_mask = otsu_img


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(c_mask['morph_ksize']))
    
    dilated_mask = cv2.dilate(current_mask, kernel, iterations=c_mask['morph_iterations'])
    
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=c_mask['morph_iterations']) 
    
    return closed_mask

def detect_hough_corners(image: np.ndarray) -> np.ndarray:
    """
    Detects the four corners of a document in the image using Hough Line Transform
    and dynamic clustering of lines.

    Args:
        image: The input color image.
        
    Returns:
        A NumPy array of shape (4, 2) representing the detected corners, or None if detection fails.
    """
    
    
    c_geo = cfg['geometry']['hough_method']
    c_pre = cfg['preprocessing']
    
    h, w = image.shape[:2]

    process_height = c_pre['resize_height']
    scale = process_height / h
    small_w = int(w * scale)
    small_img = cv2.resize(image, (small_w, process_height))
    
    mask = get_processing_mask(small_img)
    

    edges = cv2.Canny(mask, c_geo['canny']['threshold1'], c_geo['canny']['threshold2'], apertureSize=c_geo['canny']['aperture_size'])

    rho = c_geo['hough_lines']['rho']
    theta = c_geo['hough_lines']['theta_resolution'] * np.pi / 180
    thresh = c_geo['hough_lines']['threshold']
    
    lines = cv2.HoughLines(edges, rho, theta, thresh)
    
    if lines is None: return None

    lines = merge_related_lines(lines)
    
    if len(lines) < 2: return None
    
    rho_ref, theta_ref = lines[0][0]
    ref_angle = np.degrees(theta_ref)
    
    group_1, group_2 = [], [] 
    
    c_clust = c_geo['clustering']
    
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        
        diff = abs(angle - ref_angle)
        diff = min(diff, 180 - diff)
        
        if diff < c_clust['parallel_threshold_deg']: 
            group_1.append(line)
        elif c_clust['perpendicular_threshold_deg_low'] < diff < c_clust['perpendicular_threshold_deg_high']:
            group_2.append(line)
            
    if len(group_1) < 2 or len(group_2) < 2: return None

    group_1.sort(key=lambda x: x[0][0])
    group_2.sort(key=lambda x: x[0][0])
    

    side_a_1, side_a_2 = group_1[0], group_1[-1]
    side_b_1, side_b_2 = group_2[0], group_2[-1]
    
    corners = []
    corners.append(compute_intersection(side_a_1, side_b_1))
    corners.append(compute_intersection(side_a_1, side_b_2))
    corners.append(compute_intersection(side_a_2, side_b_1))
    corners.append(compute_intersection(side_a_2, side_b_2))
    
    corners = [pt for pt in corners if pt is not None]
    
    if len(corners) != 4: return None

    corners = np.array(corners, dtype="float32")
    corners = order_points(corners)

    corners = corners * (1.0 / scale)
    
    return corners