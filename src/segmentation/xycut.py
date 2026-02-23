"""
Module: xycut

Description: 
    Recursive X-Y Cut algorithm for sorting document elements.

Functions:
    - `RecursiveXYSort`: Class that implements the recursive X-Y cut sorting algorithm.
    - `sort(elements, image_w, image_h)`: Public method to sort elements based on their bounding boxes and image dimensions.
"""

import numpy as np

class RecursiveXYSort:
    def __init__(self, debug=False):
        self.debug = debug

    def _get_projection_gaps(self, boxes, axis, bounds, threshold):
        """
        Finds gaps (whitespace) along a specific axis (0=x, 1=y).
        Returns a list of split points.
        
        Args:
            boxes: List of elements.
            axis: 0 for X-axis (Vertical cuts), 1 for Y-axis (Horizontal cuts).
            bounds: Current bounding box of the area (x1, y1, x2, y2).
            threshold: Minimum pixel gap size to consider a valid split.
        """
        if not boxes:
            return []

        if axis == 0:
            intervals = [(b['bbox'][0], b['bbox'][2]) for b in boxes]
        else:
            intervals = [(b['bbox'][1], b['bbox'][3]) for b in boxes]

        intervals.sort(key=lambda x: x[0])

        merged = []
        if intervals:
            curr_start, curr_end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start < curr_end:  
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        gaps = []
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i+1][0]
            gap_size = gap_end - gap_start
            
            if gap_size > threshold: 
                mid_point = (gap_start + gap_end) / 2
                gaps.append(mid_point)

        return gaps

    def _recursive_sort(self, boxes, bounds, thresholds):
        """
        Recursive function to split boxes into groups.
        Priority: X-Cuts (Columns) -> Y-Cuts (Rows).
        
        Args:
            boxes: Elements to sort.
            bounds: Tuple (x1, y1, x2, y2).
            thresholds: Tuple (x_threshold, y_threshold).
        """
        if len(boxes) <= 1:
            return boxes

        x1, y1, x2, y2 = bounds
        x_thresh, y_thresh = thresholds

        x_cuts = self._get_projection_gaps(boxes, axis=0, bounds=(x1, x2), threshold=x_thresh)
        
        if x_cuts:
            groups = [[] for _ in range(len(x_cuts) + 1)]
            for box in boxes:
                cx = (box['bbox'][0] + box['bbox'][2]) / 2
                placed = False 
                for i, cut in enumerate(x_cuts):
                    if cx < cut:
                        groups[i].append(box)
                        placed = True
                        break
                if not placed:
                    groups[-1].append(box)

            sorted_result = []
            for group in groups:
                if group:
                    sorted_result.extend(self._recursive_sort(group, bounds, thresholds))
            return sorted_result

        y_cuts = self._get_projection_gaps(boxes, axis=1, bounds=(y1, y2), threshold=y_thresh)
        
        if y_cuts:
            groups = [[] for _ in range(len(y_cuts) + 1)]
            for box in boxes:
                cy = (box['bbox'][1] + box['bbox'][3]) / 2
                placed = False
                for i, cut in enumerate(y_cuts):
                    if cy < cut:
                        groups[i].append(box)
                        placed = True
                        break
                if not placed:
                    groups[-1].append(box)
            
            sorted_result = []
            for group in groups:
                if group:
                    sorted_result.extend(self._recursive_sort(group, bounds, thresholds))
            return sorted_result

        return sorted(boxes, key=lambda b: b['bbox'][1] * 10000 + b['bbox'][0])

    def sort(self, elements, image_w, image_h):
        """
        Public entry point.
        Calculates dynamic thresholds based on image size.
        """
        if not elements:
            return []
            
        bounds = (0, 0, image_w, image_h)
        
        x_threshold = max(1.0, image_w * 0.002) 
        y_threshold = max(1.0, image_h * 0.001)
        
        thresholds = (x_threshold, y_threshold)
        
        if self.debug:
            print(f"DEBUG: XYCut Thresholds -> X: {x_threshold:.2f}px, Y: {y_threshold:.2f}px")

        return self._recursive_sort(elements, bounds, thresholds)