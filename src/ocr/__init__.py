"""
Module: src/ocr/engine.py
Description: Handles text extraction (OCR) and element cropping for PDF reconstruction.
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
        
        # Setup Tesseract Path for Windows users
        if self.config['tesseract_cmd']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            
        self.padding = self.config['padding']
        self.ocr_classes = set(self.config['actions']['ocr'])
        self.image_classes = set(self.config['actions']['image'])

    def preprocess_for_ocr(self, image_crop):
        """
        Enhance image specifically for OCR (Binarization).
        """
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        # Otsu's thresholding is usually best for text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def clean_text(self, text):
        """
        Remove artifacts and excess whitespace.
        """
        if not text: return ""
        # Remove pipe characters | often confused with I or l in borders
        text = text.replace('|', '') 
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_padded_crop(self, image, bbox):
        """
        Crops the image with padding, ensuring we don't go out of bounds.
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Apply padding
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w_img, x2 + self.padding)
        y2 = min(h_img, y2 + self.padding)

        return image[y1:y2, x1:x2], [x1, y1, x2, y2]

    def process_layout(self, original_image, layout_elements, output_dir="data/processed/crops"):
        """
        Main function to process layout elements and generate content.
        Args:
            original_image: The full image (color).
            layout_elements: List of dicts from Phase II (sorted).
            output_dir: Where to save table/figure crops.
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_data = []

        print(f"--- Starting OCR Processing ({len(layout_elements)} elements) ---")

        for i, element in enumerate(layout_elements):
            label = element['label']
            bbox = element['bbox']
            conf = element.get('confidence', 0.0)
            
            # 1. Get Padded Crop
            crop, padded_bbox = self.get_padded_crop(original_image, bbox)
            
            content = None
            content_type = None
            file_path = None

            # 2. Logic: Text Extraction vs Image Saving
            if label in self.ocr_classes:
                # Process as Text
                content_type = "text"
                ocr_ready_img = self.preprocess_for_ocr(crop)
                
                # Configure Tesseract (Page Segmentation Mode 6 = assume a block of text)
                custom_config = r'--oem 3 --psm 6' 
                raw_text = pytesseract.image_to_string(ocr_ready_img, config=custom_config, lang=self.config['lang'])
                content = self.clean_text(raw_text)

            elif label in self.image_classes:
                # Process as Image (Table/Figure)
                content_type = "image"
                filename = f"element_{i:03d}_{label}.jpg"
                file_path = os.path.join(output_dir, filename)
                cv2.imwrite(file_path, crop)
                content = file_path # Content is the path to the image
            
            # 3. Compile Data
            item = {
                "id": i,
                "label": label,
                "type": content_type,
                "bbox": padded_bbox, # Use padded coords for PDF placement
                "content": content,
                "confidence": conf
            }
            processed_data.append(item)
            
            # Debug print
            preview = content[:30] + "..." if content_type == "text" and content else str(file_path)
            print(f"[{i}] {label}: {preview}")

        return processed_data

    def save_to_json(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {filepath}")