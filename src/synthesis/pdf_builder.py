"""
Module: src/synthesis/pdf_builder.py
Description: Reconstructs the document PDF with Dynamic Font Scaling.
"""

import os
import math
from fpdf import FPDF
from src.utils.config import cfg

class PDFReconstructor:
    def __init__(self):
        self.config = cfg['synthesis']
        self.styles = self.config['styles']
        
        # A4 Dimensions in mm
        self.PAGE_W = 210
        self.PAGE_H = 297
        self.MARGIN = 10

    def _get_scale_factor(self, img_width, img_height):
        scale_x = self.PAGE_W / img_width
        scale_y = self.PAGE_H / img_height
        return min(scale_x, scale_y)

    def _fit_text_to_box(self, pdf, text, w_mm, h_mm, font_family, style, start_size):
        """
        Iteratively reduces font size until text fits within w_mm * h_mm.
        Returns: (optimal_size, calculated_line_height)
        """
        size = start_size
        min_size = 6  # Don't go smaller than this or it's unreadable
        line_height_factor = 1.2 # Standard spacing
        
        # 0.3527 converts Points (pt) to Millimeters (mm)
        pt_to_mm = 0.3527
        
        while size >= min_size:
            pdf.set_font(font_family, style, size)
            
            # 1. Calculate length of text in one long line (in mm)
            text_width_mm = pdf.get_string_width(text)
            
            # 2. Estimate how many lines this text will wrap into
            # We add 1 buffer line for safety
            num_lines = math.ceil(text_width_mm / w_mm)
            
            # 3. Calculate total vertical height required
            one_line_height_mm = size * pt_to_mm * line_height_factor
            total_height_mm = num_lines * one_line_height_mm
            
            # 4. Check if it fits
            # We allow a small tolerance (1.1x) because PDF rendering isn't pixel perfect
            if total_height_mm <= (h_mm * 1.1): 
                return size, one_line_height_mm
            
            # If not, reduce size and try again
            size -= 0.5

        return min_size, (min_size * pt_to_mm * line_height_factor)

    def generate(self, original_image_shape, layout_data, output_path):
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        
        font_name = self.config.get('font', 'Helvetica')
        font_path = self.config.get('font_path', '')
        
        # --- Register Fonts (Regular, Bold, Italic mapped to same file) ---
        if font_path and os.path.exists(font_path):
            pdf.add_font(font_name, style="", fname=font_path)
            pdf.add_font(font_name, style="B", fname=font_path)
            pdf.add_font(font_name, style="I", fname=font_path)
            pdf.add_font(font_name, style="BI", fname=font_path)
            print(f"Loaded Unicode font: {font_name}")
        else:
            print(f"Warning: Font not found. Using Standard.")
            
        pdf.add_page()
        pdf.set_auto_page_break(auto=False)

        img_h, img_w = original_image_shape[:2]
        scale = self._get_scale_factor(img_w, img_h)

        print(f"--- Generating PDF (Scale: {scale:.4f}) ---")

        for element in layout_data:
            label = element['label']
            content_type = element['type']
            content = element['content']
            bbox = element['bbox']
            
            # Convert Coordinates
            x1_px, y1_px, x2_px, y2_px = bbox
            x_mm = x1_px * scale
            y_mm = y1_px * scale
            w_mm = (x2_px - x1_px) * scale
            h_mm = (y2_px - y1_px) * scale

            # --- TEXT HANDLING ---
            if content_type == "text":
                if not content: continue
                
                # Get preferred style from config
                style_cfg = self.styles.get(label, self.styles['Text'])
                preferred_size = style_cfg['size']
                font_style = style_cfg['style']
                
                # --- NEW: Dynamic Fitting ---
                # Calculate optimal font size to fit in the box
                optimal_size, line_height = self._fit_text_to_box(
                    pdf, content, w_mm, h_mm, font_name, font_style, preferred_size
                )
                
                # Set the optimized font
                pdf.set_xy(x_mm, y_mm)
                try:
                    pdf.set_font(font_name, style=font_style, size=optimal_size)
                except:
                    pdf.set_font(font_name, style="", size=optimal_size)

                # Use MultiCell with calculated line height
                pdf.multi_cell(w=w_mm, h=line_height, txt=content, align='J') 
                # align='J' (Justify) makes it look more like a real document

            # --- IMAGE HANDLING ---
            elif content_type == "image":
                if not os.path.exists(content): continue
                try:
                    pdf.image(content, x=x_mm, y=y_mm, w=w_mm, h=h_mm)
                except Exception as e:
                    print(f"Error adding image: {e}")

        pdf.output(output_path)
        print(f"PDF Saved: {output_path}")
        return output_path