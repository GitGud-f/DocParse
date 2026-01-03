"""
Module: geometry

Description:
Geometry-related functions for document scanning and preprocessing.
Includes corner detection and perspective transformation.

Functions:
    - order_points: Orders four points in a consistent manner.
    - four_point_transform: Applies perspective transform to get a top-down view.   
    - detect_document_corners: Robustly detects document corners using morphological operations.
"""

import cv2
import numpy as np
import imutils 

from src.utils.config import cfg

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders the four points of a quadrilateral into a consistent order.
    The order is top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: A NumPy array of shape (4, 2) representing the four points
        
    Returns:
        A NumPy array of shape (4, 2) with points ordered such that the first entry is the top-left,
        the second is the top-right, the third is the bottom-right, and the fourth is the bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left

    return rect

def get_projection_dimensions(pts: np.ndarray) -> tuple:
    """
    Calculates the width and height of the document 'as it would be' 
    after perspective transform.
    
    Args: 
        pts: A NumPy array of shape (4, 2) representing the four points of a quadrilateral.
        
    Returns:
        width: The width of the projected document.
        height: The height of the projected document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    return maxWidth, maxHeight

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to obtain a "top-down" view of the image.
    
    Args:
        image: The input image.
        pts: A NumPy array of four points representing the corners of the document.
        
    Returns:
        The transformed image.
    """

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    width, height = get_projection_dimensions(pts)

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def detect_document_corners(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Detects the four corners of a document in the image using contour detection
    and morphological operations. Refines corner positions to sub-pixel accuracy.
    
    Args:
        image: The input color image.
        debug: If True, will output intermediate images for debugging.
        
    Returns:
        A NumPy array of shape (4, 2) representing the detected corners,    
    """

    c_geo = cfg['geometry']['blob_method']
    c_pre = cfg['preprocessing']
    
    h, w = image.shape[:2]
    process_height = c_pre['resize_height']
    ratio = h / process_height
    small_image = imutils.resize(image, height=process_height)

    if len(small_image.shape) == 3:
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = small_image
        
    blurred = cv2.GaussianBlur(gray, tuple(c_geo['gaussian_blur']['ksize']), c_geo['gaussian_blur']['sigma'])
    
    edged = cv2.Canny(blurred, c_geo['canny']['threshold1'], c_geo['canny']['threshold2'])


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(c_geo['morphology']['kernel_size']))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=c_geo['morphology']['close_iterations'])
    dilated = cv2.dilate(closed, kernel, iterations=c_geo['morphology']['dilate_iterations'])

    # 2. Find Contours
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts: return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:c_geo['contours']['max_candidates']]
    best_corners = None
    best_score = -1
    TARGET_AR = c_geo['contours']['target_ar']
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, c_geo['contours']['epsilon_factor'] * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            w_proj, h_proj = get_projection_dimensions(pts)
            if w_proj == 0 or h_proj == 0: continue
            
            current_ar = max(w_proj, h_proj) / min(w_proj, h_proj)
            ar_diff = abs(current_ar - TARGET_AR)
            
            area = cv2.contourArea(c)
            norm_area = area / (small_image.shape[0] * small_image.shape[1])
            
            if norm_area < c_geo['contours']['min_area_ratio']: continue
                
            if c_geo['contours']['ar_tolerance_low'] or current_ar > c_geo['contours']['ar_tolerance_high']: 
                score_penalty = 0.5
            else: 
                score_penalty = 0
            
            score = (0.7 * norm_area) + (0.3 * (1 - ar_diff)) - score_penalty
            
            if score > best_score:
                best_score = score
                best_corners = approx

    if best_corners is None: return None


    best_corners = best_corners.astype("float32")
    best_corners *= ratio
    
    if len(image.shape) == 3:
        orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = image

    c_sub = c_geo['subpixel_refinement']
    
    # Criteria: Stop after 40 iterations or if corner moves less than 0.001 pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, c_sub['max_iterations'], c_sub['epsilon'])
    

    win_size = tuple(c_sub['window_size'])
    zero_zone = tuple(c_sub['zero_zone'])
    
    refined_corners = best_corners.reshape(-1, 1, 2)
    
    refined_corners = cv2.cornerSubPix(orig_gray, refined_corners, win_size, zero_zone, criteria)

    return refined_corners.reshape(4, 2)