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

def get_projection_dimensions(pts):
    """
    Calculates the width and height of the document 'as it would be' 
    after perspective transform.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of the new image (max distance between bottom-right/left or top-right/left)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height
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

    # Compute the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the newly warped large image
    return warped

def detect_document_corners(image, debug=False):
    """
    Detects corners prioritizing the largest contour that closely matches 
    A4 aspect ratio (approx 1.414).
    """
    # 1. Resize and Preprocess (Same as before)
    h, w = image.shape[:2]
    process_height = 800
    ratio = h / process_height
    small_image = imutils.resize(image, height=process_height)

    if len(small_image.shape) == 3:
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = small_image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aggressive Edge Detection and Morphology
    edged = cv2.Canny(blurred, 75, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2) # Fill holes
    dilated = cv2.dilate(closed, kernel, iterations=1) # Connect gaps

    if debug:
        cv2.imshow("Debug Edges", dilated)
        cv2.waitKey(0)

    # 2. Find Contours
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return None

    # Get top 10 largest contours to analyze
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    best_corners = None
    best_score = -1
    
    # A4 Aspect Ratio is sqrt(2) approx 1.414
    TARGET_AR = 1.414 
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            
            # Unpack points
            pts = approx.reshape(4, 2)
            
            # Get the dimensions this shape would have if unwarped
            w_proj, h_proj = get_projection_dimensions(pts)
            
            # Calculate Aspect Ratio (always > 1, i.e., long_side / short_side)
            if w_proj == 0 or h_proj == 0: continue
            
            current_ar = max(w_proj, h_proj) / min(w_proj, h_proj)
            
            # Calculate deviation from A4
            # 0.0 means perfect match, higher means worse
            ar_diff = abs(current_ar - TARGET_AR)
            
            # --- SCORING LOGIC ---
            # We want: High Area AND Low AR Difference.
            # Area is dominant, but AR breaks ties between similar shapes.
            
            area = cv2.contourArea(c)
            total_pixels = small_image.shape[0] * small_image.shape[1]
            norm_area = area / total_pixels
            
            # If the shape is tiny (less than 5% of screen), ignore it
            if norm_area < 0.05:
                continue
                
            # If the aspect ratio is WAY off (like a long ruler > 2.5 or a square < 1.1)
            # Penalize it heavily. A4 is 1.41. 
            # Allow flexible range [1.2, 1.8] to account for extreme perspective tilt.
            if current_ar < 1.1 or current_ar > 2.0:
                 score_penalty = 0.5 # Heavy penalty
            else:
                 score_penalty = 0
            
            # Score formula: 
            # 70% importance on Area, 30% importance on Aspect Ratio
            # (1 - ar_diff) makes sure closer to A4 gives higher score
            score = (0.7 * norm_area) + (0.3 * (1 - ar_diff)) - score_penalty
            
            if debug:
                print(f"Area: {norm_area:.2f}, AR: {current_ar:.2f}, Score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_corners = approx

    if best_corners is None:
        return None

    # 3. Rescale to original size
    best_corners = best_corners.astype("float32")
    best_corners *= ratio
    
    return best_corners.reshape(4, 2)