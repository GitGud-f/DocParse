"""
Module: inference
Description: 
    Loads the trained YOLOv8 Layout model and performs inference.
    - Supports all 11 DocLayNet classes.
    - Implements Arabic-friendly sorting (Top-to-Bottom, Right-to-Left).
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
from src.utils.config import cfg

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = cfg['segmentation']['model'].get('weights', 'models/weights/best_layout.pt')
        
        # Fallback if config returns None
        if not weights_path: 
             weights_path = 'models/weights/best_layout.pt'

        logger.info(f"Loading Layout Model from: {weights_path}")
        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
            
        self.classes = cfg['segmentation']['dataset']['class_names']
        self.conf_threshold = cfg['segmentation']['model']['conf_threshold']

    def predict(self, image: np.ndarray):
        """
        Run inference and return elements sorted for Arabic reading order.
        """
        results = self.model(image, imgsz=1280, conf=self.conf_threshold, verbose=False)[0]
        layout_elements = []
        
        h, w = image.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Safety check if model predicts a class ID outside our config list
            label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            
            # Extract crop
            y1_c, y2_c = max(0, y1), min(h, y2)
            x1_c, x2_c = max(0, x1), min(w, x2)
            crop = image[y1_c:y2_c, x1_c:x2_c]
            
            element = {
                "class_id": cls_id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop": crop
            }
            layout_elements.append(element)
            
        sorted_elements = self._sort_reading_order(layout_elements, h, w)
        
        return sorted_elements

    def _sort_reading_order(self, elements, image_height, image_width):
    
        left_col = []
        right_col = []
        mid_x = image_width / 2
        
        for el in elements:
            x1, y1, x2, y2 = el['bbox']
            center_x = (x1 + x2) / 2
            
            # Simple heuristic for 2-column papers
            if center_x < mid_x: left_col.append(el)
            else: right_col.append(el)
            
        # 2. Sort each column Top-to-Bottom
        left_col.sort(key=lambda x: x['bbox'][1])
        right_col.sort(key=lambda x: x['bbox'][1])
        
        return left_col + right_col

    def visualize(self, image, elements):
        """
        Draws bounding boxes with specific colors for DocLayNet classes.
        """
        vis_img = image.copy()
        
        # Extended Color Palette (BGR) for 11 Classes
        colors = {
            "Text": (0, 255, 0),           # Green
            "Title": (0, 0, 255),          # Red
            "Section-header": (0, 69, 255),# Orange-Red
            "Page-header": (128, 128, 128),# Gray
            "Page-footer": (128, 128, 128),# Gray
            "Table": (255, 0, 0),          # Blue
            "Picture": (255, 0, 255),      # Magenta
            "List-item": (255, 255, 0),    # Cyan
            "Caption": (0, 215, 255),      # Gold/Yellow
            "Footnote": (128, 0, 128),     # Purple
            "Formula": (203, 192, 255)     # Pinkish
        }
        
        for el in elements:
            x1, y1, x2, y2 = el['bbox']
            label = el['label']
            conf = el['confidence']
            
            # Default to white if label not found in colors
            color = colors.get(label, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label with background for readability
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Optional: Draw reading order index
            # This helps verify your RTL sorting logic
            idx = elements.index(el) + 1
            cv2.circle(vis_img, (x1, y1), 10, (0,0,0), -1)
            cv2.putText(vis_img, str(idx), (x1-4, y1+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            
        return vis_img