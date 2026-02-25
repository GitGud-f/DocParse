"""
Module: pdf_builder

Description: Advanced PDF Reconstruction using PyMuPDF.

Features: 
- Dynamic Font Scaling with Overflow Protection.
- Native Table Drawing from 2D Array Data.
"""

import os
import fitz  # PyMuPDF
import numpy as np
from src.utils.config import cfg

class PDFReconstructor:
    def __init__(self):
        self.config = cfg['synthesis']
        self.styles = self.config['styles']
        
        # A4 Dimensions in Points (standard for PDF canvas)
        self.PAGE_W = self.config.get('page_width', 595.28)
        self.PAGE_H = self.config.get('page_height', 841.89)

    def _get_scale_factor(self, img_width, img_height):
        scale_x = self.PAGE_W / img_width
        scale_y = self.PAGE_H / img_height
        return min(scale_x, scale_y)

    def _calculate_dynamic_font_size(self, text, box_height_pt):
        if not text: return 11
        num_lines = text.count('\n') + 1
        line_height_pt = box_height_pt / num_lines
        calculated_size = line_height_pt * 0.8 # Slightly safer multiplier
        return max(4.0, min(calculated_size, 40.0))

    def _safe_insert_text(self, page, rect, text, start_size, fontname, align=0):
        """
        Attempts to insert text into a rect. If it overflows, it iteratively
        shrinks the font size until it fits perfectly.
        """
        if not text.strip(): return
        
        size = start_size
        rc = -1 # rc < 0 means text didn't fit in the box
        
        while rc < 0 and size >= 4.0:
            # insert_textbox returns the height of the text if successful, or -1 if it failed
            rc = page.insert_textbox(
                rect, 
                text, 
                fontsize=size, 
                fontname=fontname, 
                align=align,
                color=(0, 0, 0)
            )
            size -= 0.5 # Shrink font size and try again
            
        # Absolute fallback: if it STILL doesn't fit (e.g., box is super tiny)
        if rc < 0:
            page.insert_text((rect.x0, rect.y0 + 8), text, fontsize=6, fontname=fontname, color=(0,0,0))

    def generate(self, original_image_shape, layout_data, output_path):
        doc = fitz.open()
        page = doc.new_page(width=self.PAGE_W, height=self.PAGE_H)

        img_h, img_w = original_image_shape[:2]
        scale = self._get_scale_factor(img_w, img_h)

        print(f"--- Generating Advanced PDF (Scale: {scale:.4f}) ---")

        for element in layout_data:
            label = element['label']
            content_type = element['type']
            content = element['content']
            bbox = element['bbox']
            
            # Coordinate Transform
            x1_px, y1_px, x2_px, y2_px = bbox
            x0 = x1_px * scale
            y0 = y1_px * scale
            x1 = x2_px * scale
            y1 = y2_px * scale
            
            rect = fitz.Rect(x0, y0, x1, y1)
            box_height_pt = y1 - y0

            # ==========================================
            # 1. TEXT ELEMENTS
            # ==========================================
            if content_type == "text" and content:
                target_fontname = self.styles.get(label, "helv")
                lines = element.get('lines_structure', [])
                ocr_scale = element.get('scale_factor', 1.0) 
                if lines and len(lines) > 0:
                    line_heights = [l['h'] for l in lines]
                    median_h_px = np.median(line_heights)
                    
                    normalized_h = median_h_px / ocr_scale
                    uniform_font_size = (normalized_h * scale) * 0.85
                    
                    uniform_font_size = max(6, min(uniform_font_size, 24))
                    
                    # ascender_offset = uniform_font_size * 0.8
                     
                    # ocr_max_w = max([l['x'] + l['w'] for l in lines])
                    # ocr_max_y = max([l['y'] + l['h'] for l in lines])
                        
                    # block_w_px = x2_px - x1_px
                    # block_h_px = y2_px - y1_px
                    
                    # scale_x_ocr = block_w_px / max(ocr_max_w, 1)
                    # scale_y_ocr = block_h_px / max(ocr_max_y, 1)
                    
                    for line in lines:
                        text = line['text']
                        # Map relative crop coords -> absolute PDF coords
                        # l_x = x0 + (line['x'] * scale_x_ocr * scale)
                        # l_y = y0 + (line['y'] * scale_y_ocr * scale)
                        # l_w = line['w'] * scale_x_ocr * scale
                        # l_h = line['h'] * scale_y_ocr * scale
                        
                        rel_x = (line['x'] / ocr_scale) * scale
                        rel_y = (line['y'] / ocr_scale) * scale
                        
                        final_x = x0 + rel_x
                        final_y = y0 + rel_y
                        
                        insert_pt = (final_x, final_y + uniform_font_size)
                        
                        page.insert_text(
                            insert_pt, 
                            text, 
                            fontsize=uniform_font_size, 
                            fontname=target_fontname
                        )
                        # Define the rect for this specific line
                        # line_rect = fitz.Rect(l_x, l_y, l_x + l_w + 50, l_y + l_h)
                        
                        # Calculate font size based on LINE height, not block height
                        # This is much more accurate
                        # font_size = l_h * 0.75 
                        # font_size = max(6, min(font_size, 14)) # Clamp reasonable sizes
                        
                        # Insert line
                        # page.insert_text((l_x, l_y + font_size), text, fontsize=font_size, fontname=target_fontname)
                      
                elif element['content']:
                    rect = fitz.Rect(x0, y0, x2_px * scale, y2_px * scale)
                    optimal_size = self._calculate_dynamic_font_size(element['content'], rect.height)
                    self._safe_insert_text(page, rect, element['content'], optimal_size, target_fontname)

                    
                # optimal_size = self._calculate_dynamic_font_size(content, box_height_pt)
                
                # self._safe_insert_text(page, rect, content, optimal_size, target_fontname, align=0)

            # ==========================================
            # 2. TABLES (Reconstructed from Data)
            # ==========================================
            elif content_type == "table":
                table_matrix = element.get('table_data')
                
                if table_matrix and len(table_matrix) > 0:
                    rows = len(table_matrix)
                    cols = max(len(r) for r in table_matrix)
                    
                    if cols > 0:
                        cell_w = rect.width / cols
                        cell_h = rect.height / rows
                        
                        for r_idx, row in enumerate(table_matrix):
                            for c_idx, cell_text in enumerate(row):
                                c_x0 = rect.x0 + (c_idx * cell_w)
                                c_y0 = rect.y0 + (r_idx * cell_h)
                                c_x1 = c_x0 + cell_w
                                c_y1 = c_y0 + cell_h
                                cell_rect = fitz.Rect(c_x0, c_y0, c_x1, c_y1)
                                
                                page.draw_rect(cell_rect, color=(0,0,0), width=0.5)
                                
                                if cell_text:
                                    pad = 2
                                    text_rect = fitz.Rect(c_x0 + pad, c_y0 + pad, c_x1 - pad, c_y1 - pad)
                                    c_size = self._calculate_dynamic_font_size(cell_text, text_rect.height *1.2)
                                    
                                    # Passing fontname="helv" cleanly
                                    self._safe_insert_text(page, text_rect, cell_text, c_size, "helv", align=1)
                else:
                    if os.path.exists(content):
                        page.insert_image(rect, filename=content)

            # ==========================================
            # 3. IMAGES (Figures, Formulas)
            # ==========================================
            elif content_type == "image":
                if os.path.exists(content):
                    page.insert_image(rect, filename=content)
                    
                    hidden_text = element.get('hidden_text', '')
                    if hidden_text:
                        hidden_size = self._calculate_dynamic_font_size(hidden_text, box_height_pt)
                        # render_mode=3 makes it invisible but perfectly searchable
                        page.insert_textbox(rect, hidden_text, fontsize=hidden_size, fontname="helv", render_mode=3)

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        
        print(f"✅ PDF Successfully Synthesized: {output_path}")
        return output_path