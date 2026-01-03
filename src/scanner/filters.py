"""
Module: filters

Description:
    A toolkit of image filtering functions for document preprocessing.
    This module focuses strictly on pixel-level transformations like contrast
    enhancement and binarization. It does not handle geometric operations.
    
Functions:
    - adaptive_thresholding: Applies adaptive thresholding to an image.
    - clahe_equalization: Applies CLAHE to enhance image contrast.
"""

import cv2
import numpy as np

from src.scanner.geometry import detect_document_corners, four_point_transform
from src.scanner.hough import detect_hough_corners
from src.utils.config import cfg

def adaptive_thresholding(image: np.ndarray, block_size: int = None, c: int = None)-> np.ndarray:
    """
    Applies adaptive thresholding to the input image.
    
    Args:
        image: The input grayscale image.
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value.
        c: Constant subtracted from the mean or weighted mean.
        
    Returns:
        The binarized image after applying adaptive thresholding.
    """
    
    if block_size is None: block_size = cfg['illumination']['adaptive_threshold']['block_size']
    if c is None: c = cfg['illumination']['adaptive_threshold']['c']
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if block_size % 2 == 0:
        block_size += 1
        
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    return thresh

def clahe_equalization(image: np.ndarray, clip_limit: float = None, tile_grid_size: tuple = None):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.
    
    Args:
        image: The input grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        The contrast-enhanced image.
    """
    
    if clip_limit is None: clip_limit = cfg['illumination']['clahe']['clip_limit']
    if tile_grid_size is None: tile_grid_size = tuple(cfg['illumination']['clahe']['tile_grid_size'])
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(gray)
    return enhanced_image
