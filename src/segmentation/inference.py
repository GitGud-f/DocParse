"""
Module: inference

Description: 
    Handles the inference pipeline for the YOLOv8 Semantic Layout Segmentation model.
    It loads the trained model, runs predictions on input images, and sorts the detected elements
    in natural reading order (Top-Down, Left-Right) to accommodate mixed single/multi-column layouts.
    
    `LayoutAnalyzer` class encapsulates the model loading, prediction, and visualization logic.
"""

import cv2
import numpy as np
import logging
from doclayout_yolo import YOLOv10

from src.utils.config import cfg
from src.segmentation.xycut import RecursiveXYSort 


# Setup module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """
    LayoutAnalyzer encapsulates the YOLOv8 model for semantic layout segmentation.
        - Loads the model with specified weights.
        - Runs inference to detect layout elements and their bounding boxes.
        - Sorts detected elements in natural reading order (Top-Down, Left-Right).
        - Provides visualization of detected elements with class-specific colors and confidence scores.
    """
    def __init__(self, weights_path=None):
        """
        Initializes the LayoutAnalyzer by loading the YOLOv8 model with specified weights.
        Args:
            weights_path: Optional path to the model weights. If None, it will use the default path
                          specified in the configuration or fallback to 'models/weights/best_layout.pt'.
        """
        if weights_path is None:
            weights_path = cfg['segmentation']['model'].get('weights', 'models/weights/best_layout.pt')
        
        if not weights_path: 
             weights_path = 'models/weights/doclayout_yolo_docstructbench_imgsz1024.pt'

        logger.info(f"Loading Layout Model from: {weights_path}")
        try:
            self.model = YOLOv10(weights_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
            
        self.classes = cfg['segmentation']['dataset']['class_names']
        self.conf_threshold = cfg['segmentation']['model']['conf_threshold']
        self.sorter = RecursiveXYSort()
        self.ignore_labels = {"Abandon"} 

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def clean_detections(self, elements):
        """
        Removes overlapping duplicates or contained boxes of the same class.
        """
        valid_indices = set(range(len(elements)))
        
        for i in range(len(elements)):
            if i not in valid_indices: continue
            
            for j in range(i + 1, len(elements)):
                if j not in valid_indices: continue
                
                boxA = elements[i]['bbox']
                boxB = elements[j]['bbox']
                iou = self.calculate_iou(boxA, boxB)
                
                # If high overlap
                if iou > 0.7: 
                    # If same class, keep higher confidence
                    if elements[i]['label'] == elements[j]['label']:
                        if elements[i]['confidence'] > elements[j]['confidence']:
                            valid_indices.remove(j)
                        else:
                            valid_indices.remove(i)
                            break 
                        
                    elif elements[i]['label'] == 'Text' and elements[j]['label'] in ['Table', 'Figure']:
                        valid_indices.remove(i)
                        break
        
        return [elements[i] for i in sorted(valid_indices)]


    def predict(self, image: np.ndarray) -> list:
        """
        Runs inference on the input image and returns a list of detected layout elements
        with their class labels, bounding boxes, confidence scores, and cropped images.
        
        Args:
            image: Input image as a NumPy array (BGR format)
            
        Returns:
            List of dicts, each containing:
                - class_id: int
                - label: str
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - crop: np.ndarray (cropped image of the detected element)
        """
        results = self.model(image, imgsz=1024, conf=self.conf_threshold, verbose=False)[0]
        layout_elements = []
        
        h, w = image.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            
            if label in self.ignore_labels:
                continue
                
            
            # y1_c, y2_c = max(0, y1), min(h, y2)
            # x1_c, x2_c = max(0, x1), min(w, x2)
            # crop = image[y1_c:y2_c, x1_c:x2_c]
            
            element = {
                "class_id": cls_id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                # "crop": crop
            }
            layout_elements.append(element)
        
        cleaned_elements = self.clean_detections(layout_elements)
          
        sorted_elements = self.sorter.sort(cleaned_elements, w, h)
        
        return sorted_elements

    def visualize(self, image: np.ndarray, elements: list) -> np.ndarray:
        """
        Draws bounding boxes with specific colors for DocLayNet classes.
        
        Args:
            image: Original image (BGR)
            elements: List of detected elements with 'bbox', 'label', and 'confidence'
            
        Returns:            
            Image with drawn bounding boxes and labels.
        """
        vis_img = image.copy()
        
        colors = {
            "Title": (0, 0, 255),          # Red
            "Text": (0, 255, 0),           # Green
            "Abandon": (128, 128, 128),    # Gray
            "Figure": (255, 0, 255),       # Magenta
            "Figure-caption": (0, 215, 255),# Gold
            "Table": (255, 0, 0),          # Blue
            "Table-caption": (0, 165, 255),# Orange
            "Table-footnote": (128, 0, 128),# Purple
            "Formula": (203, 192, 255),    # Pink
            "Formula-caption": (255, 255, 0)# Cyan
        }
        
        for el in elements:
            x1, y1, x2, y2 = el['bbox']
            label = el['label']
            conf = el['confidence']
            
            
            color = colors.get(label, (255, 255, 255))
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            idx = elements.index(el) + 1
            cv2.circle(vis_img, (x1, y1), 10, (0,0,0), -1)
            cv2.putText(vis_img, str(idx), (x1-4, y1+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            
        return vis_img