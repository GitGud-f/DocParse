import numpy as np
from shapely.geometry import Polygon
# import Levenshtein
# import re

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

# def clean_text(text: str) -> str:
#     """
#     Normalizes text for fair comparison.
#     Removes extra whitespace, newlines, and converts to lowercase.
#     """
#     if not text: return ""
#     # Replace newlines and multiple spaces with single space
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip().lower()

# def calculate_cer(reference: str, hypothesis: str) -> float:
#     """
#     Calculates Character Error Rate (CER).
#     CER = (Substitutions + Deletions + Insertions) / Total Characters
    
#     Args:
#         reference: The ground truth text.
#         hypothesis: The OCR output text.
        
#     Returns:
#         float: Lower is better (0.0 is perfect).
#     """
#     ref = clean_text(reference)
#     hyp = clean_text(hypothesis)
    
#     if len(ref) == 0:
#         return 1.0 if len(hyp) > 0 else 0.0
        
#     distance = Levenshtein.distance(ref, hyp)
#     return distance / len(ref)

# def calculate_wer(reference: str, hypothesis: str) -> float:
#     """
#     Calculates Word Error Rate (WER).
#     """
#     ref = clean_text(reference).split()
#     hyp = clean_text(hypothesis).split()
    
#     if len(ref) == 0:
#         return 1.0 if len(hyp) > 0 else 0.0
        
#     # Levenshtein on lists treats elements (words) as tokens
#     distance = Levenshtein.distance(ref, hyp)
#     return distance / len(ref)