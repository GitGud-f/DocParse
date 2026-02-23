"""
Module: src/ocr/table_parser.py
Description: Advanced Grid Extraction using Line Subtraction and Projection Profiles.
Includes heavy visual debugging to diagnose extraction failures.
"""
import cv2
import numpy as np
import os

class TableParser:
    def __init__(self, tesseract_config, lang='eng', debug_dir="data/output/debug_tables"):
        self.tesseract_config = tesseract_config
        self.lang = lang
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

    def _get_projection_splits(self, mask, axis, min_gap=3):
        """
        Calculates the projection profile along the given axis and finds gaps.
        axis=0: Vertical projection (finds columns)
        axis=1: Horizontal projection (finds rows)
        """
        projection = np.sum(mask, axis=axis)
        
        # A gap is where the sum of pixels is very low (less than 3 pixels of noise)
        whitespace_mask = projection <= (255 * 3) 

        splits = []
        in_gap = False
        gap_start = 0

        for i, is_white in enumerate(whitespace_mask):
            if is_white and not in_gap:
                in_gap = True
                gap_start = i
            elif not is_white and in_gap:
                in_gap = False
                gap_end = i
                
                if (gap_end - gap_start) >= min_gap:
                    # Cut exactly in the middle of the whitespace gap
                    split_point = (gap_start + gap_end) // 2
                    splits.append(split_point)

        max_dim = mask.shape[0 if axis == 1 else 1]
        final_splits = [0] + splits + [max_dim]
        
        # Deduplicate splits that are too close (prevents 1px wide cells)
        filtered_splits = [final_splits[0]]
        for s in final_splits[1:]:
            if s - filtered_splits[-1] > 5:
                filtered_splits.append(s)

        return filtered_splits

    def extract_structure_projection(self, binary_crop, table_id):
        # 0. Invert: text becomes white (255), background becomes black (0)
        inverted = cv2.bitwise_not(binary_crop)
        cv2.imwrite(os.path.join(self.debug_dir, f"{table_id}_01_inverted.jpg"), inverted)

        # 1. Isolate horizontal and vertical lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(inverted.shape[1] // 4, 15), 1))
        h_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(inverted.shape[0] // 4, 15)))
        v_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel, iterations=2)

        lines_mask = cv2.add(h_lines, v_lines)
        # cv2.imwrite(os.path.join(self.debug_dir, f"{table_id}_02_lines.jpg"), lines_mask)

        # 2. Subtract lines to leave ONLY text
        text_only = cv2.subtract(inverted, lines_mask)
        # cv2.imwrite(os.path.join(self.debug_dir, f"{table_id}_03_text_only.jpg"), text_only)

        # 3. Dilate text slightly to merge characters in the same word
        # Using a very small kernel so we don't accidentally merge adjacent columns!
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        text_thick = cv2.dilate(text_only, dilate_kernel, iterations=1)
        # cv2.imwrite(os.path.join(self.debug_dir, f"{table_id}_04_text_thick.jpg"), text_thick)

        # 4. Find splits
        y_splits = self._get_projection_splits(text_thick, axis=1, min_gap=3)
        x_splits = self._get_projection_splits(text_thick, axis=0, min_gap=5)

        # # --- DEBUG DRAWING: Draw the detected grid over the original image ---
        # debug_grid = cv2.cvtColor(binary_crop, cv2.COLOR_GRAY2BGR)
        # for x in x_splits:
        #     cv2.line(debug_grid, (x, 0), (x, debug_grid.shape[0]), (0, 0, 255), 1) # Red columns
        # for y in y_splits:
        #     cv2.line(debug_grid, (0, y), (debug_grid.shape[1], y), (255, 0, 0), 1) # Blue rows
        # cv2.imwrite(os.path.join(self.debug_dir, f"{table_id}_05_final_grid.jpg"), debug_grid)

        return x_splits, y_splits

    def parse(self, color_crop, binary_crop, ocr_func, table_id="table_0"):
        x_splits, y_splits = self.extract_structure_projection(binary_crop, table_id)

        if len(x_splits) < 2 or len(y_splits) < 2:
            return None

        table_data = []
        cells_dir = os.path.join(self.debug_dir, f"{table_id}_cells")
        os.makedirs(cells_dir, exist_ok=True)

        for row_idx in range(len(y_splits) - 1):
            y1, y2 = y_splits[row_idx], y_splits[row_idx + 1]
            row_data = []
            
            for col_idx in range(len(x_splits) - 1):
                x1, x2 = x_splits[col_idx], x_splits[col_idx + 1]
                
                # Add 2px padding so we don't cut the edges of characters
                pad = 2
                c_y1, c_y2 = max(0, y1+pad), min(binary_crop.shape[0], y2-pad)
                c_x1, c_x2 = max(0, x1+pad), min(binary_crop.shape[1], x2-pad)
                
                cell_crop = binary_crop[c_y1:c_y2, c_x1:c_x2]
                
                if cell_crop.shape[0] < 5 or cell_crop.shape[1] < 5:
                    row_data.append("")
                    continue
                
                # Check for empty cell (black text on white bg)
                black_pixels = np.sum(cell_crop == 0)
                if black_pixels < 15:
                    row_data.append("")
                    continue

                # Run OCR
                text = ocr_func(cell_crop)
                row_data.append(text)
                
                # DEBUG: Save individual cell crops that actually have text
                # if text.strip():
                #     cv2.imwrite(os.path.join(cells_dir, f"r{row_idx}_c{col_idx}_{text[:5]}.jpg"), cell_crop)
                
            if any(cell.strip() != "" for cell in row_data):
                table_data.append(row_data)

        # Normalize matrix length
        if table_data:
            max_cols = max(len(row) for row in table_data)
            for row in table_data:
                while len(row) < max_cols:
                    row.append("")
            return table_data
            
        return None