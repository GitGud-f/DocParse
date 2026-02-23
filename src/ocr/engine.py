"""
Module: engine

Description: 
    This module implements the OCREngine class, which performs context-aware OCR on document images. It uses Tesseract for text extraction and applies dynamic configurations based on semantic labels from layout analysis. The engine also includes heuristic post-processing to clean OCR output and an optional spellchecking step for improved accuracy. For visual elements like tables and figures, it extracts both the image crop and any hidden text layer, making them searchable in the final output. The processed data is structured in a JSON format for downstream use in PDF generation and other applications.

"""
import cv2
import numpy as np
import pytesseract
import json
import os
import re

from src.utils.config import cfg
from src.ocr.table_parser import TableParser

try:
    from spellchecker import SpellChecker
    print("Spellchecker module found. Spellchecking enabled.")
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False
    print("Warning: 'pyspellchecker' not found. Spellchecking disabled. Run `pip install pyspellchecker`.")

class OCREngine:
    def __init__(self):
        self.config = cfg['ocr']
        
        if self.config['tesseract_cmd']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            
        self.padding = self.config['padding']
        self.lang = self.config['lang']
        self.min_height = self.config['preprocessing']['min_height_for_ocr']
        self.upscale_factor = self.config['preprocessing']['upscale_factor']
        self.enable_spellcheck = self.config['postprocessing']['enable_spellcheck'] and SPELLCHECK_AVAILABLE
        
        self.ocr_classes = set(self.config['actions']['ocr'])
        self.image_classes = set(self.config['actions']['image'])
        
        if self.enable_spellcheck:
            self.spell = SpellChecker(distance=1) 

    def get_tesseract_config(self, label):
        """
        Dynamically adjusts Page Segmentation Mode (PSM) based on Semantic Label.
        """
        base_config = r'--oem 3' 
        
        if label in ["Title", "Section-header"]:
            return f'{base_config} --psm 7'
            
        elif label in ["Figure-caption", "Table-caption", "Table-footnote", "Formula-caption"]:
            return f'{base_config} --psm 6'
            
        elif label == "Text":
            return f'{base_config} --psm 3'
            
        elif label in ["Table", "Figure"]:
            return f'{base_config} --psm 4'
            

        return f'{base_config} --psm 6'

    def upscale_crop_if_needed(self, crop_img):
        """
        Upscales small image crops (like footnotes) to meet DPI requirements for Tesseract.
        """
        h, w = crop_img.shape[:2]
        if h < self.min_height:
            new_w = int(w * self.upscale_factor)
            new_h = int(h * self.upscale_factor)

            upscaled = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            

            if len(upscaled.shape) == 2:
                _, upscaled = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)
            return upscaled
        return crop_img

    def clean_text(self, text, label):
        """
        Heuristic post-processing to clean Tesseract output.
        """
        if not text: return ""
        

        text = text.replace('|', 'I') 
        text = re.sub(r'\s+', ' ', text) 
        text = text.strip()
        


        text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'O', text) # '0' between letters -> 'O'
        text = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', text)       # 'O' between numbers -> '0'
        text = re.sub(r'(?<=[a-zA-Z])5(?=[a-zA-Z])', 'S', text) # '0' between letters -> 'O'
        text = re.sub(r'(?<=[0-9])S(?=[0-9])', '5', text)       # 'O' between numbers -> '0'
        

        if self.enable_spellcheck and label == "Text" and len(text) > 3:
            words = text.split()
            corrected_words = []
            for word in words:

                if word.isalpha():
                    corr = self.spell.correction(word)
                    corrected_words.append(corr if corr else word)
                else:
                    corrected_words.append(word)
            text = " ".join(corrected_words)


        if label == "Title":
            text = text.title()
            
        return text

    def get_padded_crop(self, image, bbox):
        """Generic cropper with padding."""
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w_img, x2 + self.padding)
        y2 = min(h_img, y2 + self.padding)

        return image[y1:y2, x1:x2], [x1, y1, x2, y2]

    def process_layout(self, color_image, binary_image, layout_elements, output_dir="data/processed/crops"):
        os.makedirs(output_dir, exist_ok=True)
        processed_data = []

        print(f"--- Starting Context-Aware Hybrid OCR ---")

        for i, element in enumerate(layout_elements):
            label = element['label']
            bbox = element['bbox']
            conf = element.get('confidence', 0.0)
            
            content = ""
            hidden_text = ""
            content_type = None
            

            custom_config = self.get_tesseract_config(label)


            if label in self.ocr_classes:
                content_type = "text"
                crop_bin, padded_bbox = self.get_padded_crop(binary_image, bbox)
                

                crop_bin = self.upscale_crop_if_needed(crop_bin)
                
                try:
                    raw_text = pytesseract.image_to_string(crop_bin, config=custom_config, lang=self.lang)
                    content = self.clean_text(raw_text, label)
                except Exception as e:
                    print(f"OCR Error on element {i} ({label}): {e}")

            elif label == "Table":
                content_type = "table"
                crop_color, padded_bbox = self.get_padded_crop(color_image, bbox)
                crop_bin, _ = self.get_padded_crop(binary_image, bbox)
                
                # 1. Save visual representation for the final PDF
                filename = f"element_{i:03d}_{label}.jpg"
                file_path = os.path.join(output_dir, filename)
                cv2.imwrite(file_path, crop_color)
                content = file_path 
                
                # 2. Try to parse the table grid
                parser = TableParser(tesseract_config=custom_config, lang=self.lang)
                
                # Callback function to handle cell OCR logic
                def cell_ocr_callback(cell_img):
                    upscaled = self.upscale_crop_if_needed(cell_img)
                    # PSM 6 or 7 is best for single cells
                    cell_config = self.get_tesseract_config("Text") 
                    raw_text = pytesseract.image_to_string(upscaled, config=cell_config, lang=self.lang)
                    return self.clean_text(raw_text, "Text")
                
                table_data = parser.parse(crop_color, crop_bin, cell_ocr_callback, table_id=f"table_{i}")
                
                # If grid parsing fails (e.g., borderless table), fallback to hidden text
                if not table_data:
                    crop_bin_hidden = self.upscale_crop_if_needed(crop_bin)
                    try:
                        raw_hidden = pytesseract.image_to_string(crop_bin_hidden, config=custom_config, lang=self.lang)
                        hidden_text = self.clean_text(raw_hidden, label)
                    except Exception as e:
                        pass
                else:
                    # Convert parsed 2D array into a formatted hidden string for the PDF later
                    hidden_text = "\n".join(["\t".join(row) for row in table_data])
                    
            elif label in self.image_classes:
                content_type = "image"
                crop_color, padded_bbox = self.get_padded_crop(color_image, bbox)
                

                filename = f"element_{i:03d}_{label}.jpg"
                file_path = os.path.join(output_dir, filename)
                cv2.imwrite(file_path, crop_color)
                content = file_path 
                


                crop_bin_hidden, _ = self.get_padded_crop(binary_image, bbox)
                crop_bin_hidden = self.upscale_crop_if_needed(crop_bin_hidden)
                
                try:
                    raw_hidden = pytesseract.image_to_string(crop_bin_hidden, config=custom_config, lang=self.lang)
                    hidden_text = self.clean_text(raw_hidden, label)
                except Exception as e:
                    print(f"Hidden OCR Error on element {i} ({label}): {e}")
            
            else:
                continue

            item = {
                "id": i,
                "label": label,
                "type": content_type,
                "bbox": padded_bbox,
                "content": content,
                "hidden_text": hidden_text, 
                "table_data": table_data if label == "Table" else None, # NEW
                "confidence": conf
            }
            processed_data.append(item)

        return processed_data

    def save_to_json(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)