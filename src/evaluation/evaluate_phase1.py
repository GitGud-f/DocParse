import cv2
import json
import os
import numpy as np
from tqdm import tqdm 
import pandas as pd   

from src.scanner.geometry import detect_document_corners, order_points
from src.scanner.hough import detect_hough_corners
from src.evaluation.metrics import calculate_iou, categorize_score
from src.utils.config import cfg

# --- Configuration ---
BENCHMARK_DIR = "data/benchmark"
GT_FILE = os.path.join(BENCHMARK_DIR, "ground_truth.json")

def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def run_evaluation(method="blob"):
    print(f"\n--- Starting Evaluation using Method: {method} ---")
    
    if not os.path.exists(GT_FILE):
        print(f"Error: Ground truth file not found at {GT_FILE}")
        return

    gt_data = load_ground_truth(GT_FILE)
    results = []
    
    total_iou = 0
    success_count = 0 # Count > 0.85
    
    for filename, coords in tqdm(gt_data.items(), desc="Evaluated"):
        image_path = os.path.join(BENCHMARK_DIR, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found.")
            continue
            
        image = cv2.imread(image_path)
        gt_corners = np.array(coords, dtype="float32")
        gt_corners = order_points(gt_corners) 
        
        start_tick = cv2.getTickCount()
        if method == "blob":
            pred_corners = detect_document_corners(image)
        else:
            pred_corners = detect_hough_corners(image)
        end_tick = cv2.getTickCount()
        time_ms = (end_tick - start_tick) / cv2.getTickFrequency() * 1000
            
        # Calculate Metric
        if pred_corners is not None:
            pred_corners = order_points(pred_corners)
            iou = calculate_iou(gt_corners, pred_corners)
        else:
            iou = 0.0
            
        # Log Result
        cat = categorize_score(iou)
        if iou > 0.85: success_count += 1
        total_iou += iou
        
        results.append({
            "Image": filename,
            "IoU": round(iou, 4),
            "Category": cat,
            "Time(ms)": round(time_ms, 2)
        })

    # Summary
    df = pd.DataFrame(results)
    avg_iou = total_iou / len(gt_data)
    accuracy = (success_count / len(gt_data)) * 100
    
    print("\n" + "="*40)
    print(f"  RESULTS SUMMARY ({method})")
    print("="*40)
    print(f"Total Images:    {len(gt_data)}")
    print(f"Average IoU:     {avg_iou:.2%}")
    print(f"Success Rate:    {accuracy:.2f}% (IoU > 0.85)")
    print(f"Avg Time/Img:    {df['Time(ms)'].mean():.2f} ms")
    print("-" * 40)
    
    # Save detailed report
    df.to_csv(f"data/output/evaluation_report_{method}.csv", index=False)
    print(f"Detailed report saved to data/output/evaluation_report_{method}.csv")
    
    # Show worst performers
    print("\nTop 3 Worst Performers:")
    print(df.sort_values(by="IoU").head(3))

if __name__ == "__main__":
    # Choose method to evaluate
    run_evaluation(method="hough")
    run_evaluation(method="blob")