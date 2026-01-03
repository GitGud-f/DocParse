import cv2
import json
import os
import glob

# Setup
IMAGE_DIR = "data/benchmark"
OUTPUT_JSON = os.path.join(IMAGE_DIR, "ground_truth.json")

points = []

def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(params['img'], (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Annotator", params['img'])

def create_ground_truth():
    gt_data = {}
    
    # Load existing if available to resume
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            gt_data = json.load(f)
            
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    print("--- Annotation Tool ---")
    print("Click 4 corners for each image.")
    print("Press 'n' to save and next image.")
    print("Press 's' to skip image.")
    print("Press 'q' to quit.")

    for filepath in files:
        filename = os.path.basename(filepath)
        if filename in gt_data:
            print(f"Skipping {filename} (already annotated)")
            continue
            
        img = cv2.imread(filepath)
        if img is None: continue
        
        # Resize for easier clicking if huge
        scale = 1.0
        h, w = img.shape[:2]
        if h > 900:
            scale = 900 / h
            img_display = cv2.resize(img, (0,0), fx=scale, fy=scale)
        else:
            img_display = img.copy()

        global points
        points = []
        
        cv2.imshow("Annotator", img_display)
        cv2.setMouseCallback("Annotator", click_event, {'img': img_display})

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'): # Next
                if len(points) == 4:
                    # Scale points back to original size
                    orig_points = [[int(p[0]/scale), int(p[1]/scale)] for p in points]
                    gt_data[filename] = orig_points
                    print(f"Saved {filename}")
                    break
                else:
                    print("Need exactly 4 points!")
            elif key == ord('s'): # Skip
                break
            elif key == ord('q'): # Quit
                with open(OUTPUT_JSON, 'w') as f:
                    json.dump(gt_data, f, indent=4)
                return

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(gt_data, f, indent=4)
    print(f"Ground truth saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    create_ground_truth()