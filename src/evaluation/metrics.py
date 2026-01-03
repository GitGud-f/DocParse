import numpy as np
from shapely.geometry import Polygon

def calculate_iou(gt_corners: np.ndarray, pred_corners: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU) for two quadrilaterals.
    
    Args:
        gt_corners: Shape (4, 2)
        pred_corners: Shape (4, 2)
        
    Returns:
        float: IoU score between 0.0 and 1.0
    """
    if pred_corners is None:
        return 0.0
        
    poly_gt = Polygon(gt_corners)
    poly_pred = Polygon(pred_corners)

    if not poly_gt.is_valid or not poly_pred.is_valid:
        return 0.0

    intersection_area = poly_gt.intersection(poly_pred).area
    
    union_area = poly_gt.union(poly_pred).area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def categorize_score(iou: float) -> str:
    if iou >= 0.95: return "Perfect"
    if iou >= 0.85: return "Good"
    if iou >= 0.70: return "Acceptable"
    return "Fail"