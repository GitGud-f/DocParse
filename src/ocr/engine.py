"""
Module: src/ocr/engine.py
"""
import cv2
import numpy as np
import pytesseract
import json
import os
import re
from src.utils.config import cfg

class OCREngine:
    def __init__(self):
        self.config = cfg['ocr']
        
        # Setup Tesseract
        if self.config['tesseract_cmd']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            
        self.padding = self.config['padding']
        self.lang = self.config['lang']
        
        # Load classes from config
        self.ocr_classes = set(self.config['actions']['ocr'])
        self.image_classes = set(self.config['actions']['image'])

    def clean_text(self, text):
        if not text: return ""
        text = text.replace('|', '') 
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_padded_crop(self, image, bbox):
        """
        Generic cropper that works on both Color (3-channel) and Binary (1-channel) images.
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Apply padding
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w_img, x2 + self.padding)
        y2 = min(h_img, y2 + self.padding)

        return image[y1:y2, x1:x2], [x1, y1, x2, y2]

    def process_layout(self, color_image, binary_image, layout_elements, output_dir="data/processed/crops"):
        """
        Args:
            color_image: The Warped Color image (for Tables/Figures).
            binary_image: The Phase I Binarized image (for Text OCR).
            layout_elements: YOLO detections.
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_data = []

        print(f"--- Starting Hybrid OCR Processing ---")

        for i, element in enumerate(layout_elements):
            label = element['label']
            bbox = element['bbox']
            conf = element.get('confidence', 0.0)
            
            content = None
            content_type = None
            
            # --- STRATEGY 1: TEXT (Use Binary Image) ---
            if label in self.ocr_classes:
                content_type = "text"
                
                # Crop from the BINARIZED image
                # No need to pre-process/threshold again!
                crop_bin, padded_bbox = self.get_padded_crop(binary_image, bbox)
                
                # Tesseract Config:
                # --psm 6: Assume a single uniform block of text.
                custom_config = r'--oem 3 --psm 6' 
                
                try:
                    raw_text = pytesseract.image_to_string(crop_bin, config=custom_config, lang=self.lang)
                    content = self.clean_text(raw_text)
                except Exception as e:
                    print(f"OCR Error on element {i}: {e}")
                    content = ""

            # --- STRATEGY 2: VISUALS (Use Color Image) ---
            elif label in self.image_classes:
                content_type = "image"
                
                # Crop from the COLOR image (we want the figure to look good)
                crop_color, padded_bbox = self.get_padded_crop(color_image, bbox)
                
                filename = f"element_{i:03d}_{label}.jpg"
                file_path = os.path.join(output_dir, filename)
                
                cv2.imwrite(file_path, crop_color)
                content = file_path # Content is the path
            
            # Fallback for unknown labels
            else:
                continue

            # Compile Data
            item = {
                "id": i,
                "label": label,
                "type": content_type,
                "bbox": padded_bbox,
                "content": content,
                "confidence": conf
            }
            processed_data.append(item)

        return processed_data

    def save_to_json(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)