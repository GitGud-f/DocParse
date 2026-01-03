"""
Module: hough

Description:
    Hough Transform-based functions for document corner detection.
    Includes line merging and intersection computation.
    !THIS IS EXPERIMENTAL!
    
Functions:
    - compute_intersection: Computes intersection point of two lines in (rho, theta) format.
    - merge_related_lines: Merges similar lines based on proximity in (rho, theta) space.
    - get_angle_diff: Calculates the smallest difference between two angles in degrees.
    - separate_line_orientations: Separates lines into two groups based on orientation.
    - get_processing_mask: Generates a binary mask for edge detection.
    - detect_hough_corners: Detects document corners using Hough Line Transform.
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
        A list of merged lines in (rho, theta) format.
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

def get_angle_diff(angle1_deg: float, angle2_deg: float ) -> float:
    """
    Calculates the smallest difference between two angles in degrees.
    Handles the 0/180 wrap-around for lines.
    
    Args:
        angle1_deg: First angle in degrees.
        angle2_deg: Second angle in degrees.
        
    Returns:
        The smallest difference between the two angles in degrees.
    """
    diff = abs(angle1_deg - angle2_deg)
    return min(diff, 180 - diff)

def separate_line_orientations(lines: list, config: dict) -> tuple:
    """
    Separates lines into two groups based on orientation using dynamic clustering.
    Args:
        lines: A list of lines in (rho, theta) format.
        config: Configuration dictionary with clustering thresholds.
    Returns:
        A tuple of two lists: (group_1_lines, group_2_lines)    
    """
    if lines is None or len(lines) < 2:
        return None, None

    clean_lines = []
    for line in lines:
        rho, theta = line[0]
        deg = np.degrees(theta) % 180
        clean_lines.append({'rho': rho, 'theta': theta, 'deg': deg, 'data': line})
    
    anchor_angle_1 = None
    anchor_angle_2 = None
    
    perp_low = config['perpendicular_threshold_deg_low']   
    perp_high = config['perpendicular_threshold_deg_high'] 

    found_pair = False
    
    for i in range(len(clean_lines)):
        angle_i = clean_lines[i]['deg']
        for j in range(i + 1, len(clean_lines)):
            angle_j = clean_lines[j]['deg']
            
            diff = get_angle_diff(angle_i, angle_j)
            
            if perp_low < diff < perp_high:
                anchor_angle_1 = angle_i
                anchor_angle_2 = angle_j
                found_pair = True
                break
        if found_pair: break
    
    if not found_pair:
        if len(clean_lines) > 0:
            anchor_angle_1 = clean_lines[0]['deg']
            anchor_angle_2 = (anchor_angle_1 + 90) % 180
        else:
            return None, None

    group_1 = []
    group_2 = []
    parallel_thresh = config['parallel_threshold_deg'] 

    for item in clean_lines:
        dist_1 = get_angle_diff(item['deg'], anchor_angle_1)
        dist_2 = get_angle_diff(item['deg'], anchor_angle_2)
        
        if dist_1 < parallel_thresh:
            group_1.append(item['data'])
        elif dist_2 < parallel_thresh:
            group_2.append(item['data'])
            
    return group_1, group_2

def get_processing_mask(image: np.ndarray) -> np.ndarray:
    """
    Generates the binary mask used for edge detection.
    Optimized to handle both Dark (Otsu) and Light (Adaptive) backgrounds,
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
    
    otsu_thresh_val, otsu_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    white_pixels = np.count_nonzero(otsu_img)
    total_pixels = otsu_img.size
    white_ratio = white_pixels / total_pixels
    
    if c_mask['otsu_white_ratio_lower'] < white_ratio < c_mask['otsu_white_ratio_upper']:
        current_mask = otsu_img
    else:
        current_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, c_mask['adaptive_block_size'], 
                                             c_mask['adaptive_c'])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(c_mask['morph_ksize']))
    dilated_mask = cv2.dilate(current_mask, kernel, iterations=c_mask['morph_iterations'])
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=c_mask['morph_iterations']) 
    
    return closed_mask

def detect_hough_corners(image: np.ndarray) -> np.ndarray:
    """
    Detects document corners using Hough Line Transform.
    
    Args:
        image: The input image in which to detect corners.
        
    Returns:
        A numpy array of shape (4, 2) containing the corner points in the order
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

    lines = cv2.HoughLines(edges, 
                           c_geo['hough_lines']['rho'], 
                           c_geo['hough_lines']['theta_resolution'] * np.pi / 180, 
                           c_geo['hough_lines']['threshold'])
    
    if lines is None: return None

    lines = merge_related_lines(lines)
    
    group_a, group_b = separate_line_orientations(lines, c_geo['clustering'])
    
    if not group_a or not group_b: return None

    group_a.sort(key=lambda x: x[0][0])
    group_b.sort(key=lambda x: x[0][0])
    
    side_a_1 = group_a[0]
    side_a_2 = group_a[-1]
    side_b_1 = group_b[0]
    side_b_2 = group_b[-1]
    
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