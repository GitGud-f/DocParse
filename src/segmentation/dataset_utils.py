
"""
Module: dataset_utils

Description:
    Utilities for downloading and formatting the DocLayNet-small dataset
    from HuggingFace into YOLOv8 format.
    
Fuctions:
    - convert_to_yolo_bbox: Converts COCO bbox format to YOLO format.
    - save_yolo_label: Saves bounding boxes and categories into a YOLO .txt file.
    - prepare_publaynet: Main function to download, convert, and save the dataset
      along with generating the dataset.yaml configuration file for YOLOv8 training.
"""

import os
import yaml
import logging
from tqdm import tqdm
from datasets import load_dataset
from src.utils.config import cfg

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_yolo_bbox(bbox: list, img_width: int, img_height: int) -> list:
    """
    Converts a bounding box from COCO format [x_min, y_min, width, height]
    to YOLO format [x_center, y_center, width, height], normalized by image dimensions.
    
    Args:
        bbox: List of [x_min, y_min, width, height]
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        List of [x_center, y_center, width, height] in YOLO format
    """
    x_min, y_min, w, h = bbox

    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return [x_center, y_center, w_norm, h_norm]

def save_yolo_label(bboxes: list, categories: list, img_width: int, img_height: int, label_path: str, mapping: dict):
    """
    Parses DocLayNet parallel lists and writes a YOLO format .txt file.
    
    Args:
        bboxes: List of [x,y,w,h]
        categories: List of int IDs
        img_width: Width of the image
        img_height: Height of the image
        label_path: Path to save the .txt file
        mapping: Dict mapping source ID to target ID
    """
    yolo_lines = []
    
    for bbox, cat_id in zip(bboxes, categories):
        if cat_id not in mapping:
            continue

        yolo_id = mapping[cat_id]

        x_c, y_c, w, h = convert_to_yolo_bbox(bbox, img_width, img_height)

        x_c, y_c = max(0.0, min(1.0, x_c)), max(0.0, min(1.0, y_c))
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))

        yolo_lines.append(f"{yolo_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def prepare_doclaynet():
    c_seg = cfg['segmentation']['dataset']
    output_dir = c_seg['training_dir']

    logger.info(f"--- Starting DocLayNet Download & Conversion ---")

    try:
        ds = load_dataset(c_seg['repo_id'])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # DocLayNet-small uses 'validation' and 'test' keys, but we map them to our logic
    splits_to_process = {
        'train': {'data': ds['train'], 'limit': c_seg['limit_train']},
        'val': {'data': ds['validation'], 'limit': c_seg['limit_val']}
    }

    mapping = c_seg['id_mapping']

    for split_name, split_info in splits_to_process.items():
        dataset_obj = split_info['data']
        limit = split_info['limit']

        img_dir = os.path.join(output_dir, "images", split_name)
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        pbar = tqdm(total=min(len(dataset_obj), limit), desc=f"Processing {split_name}")

        for i, sample in enumerate(dataset_obj):
            if i >= limit:
                break

            try:
                image = sample['image']
                if image.mode != "RGB":
                    image = image.convert("RGB")

                img_w, img_h = image.size

                bboxes = sample.get('bboxes_block', [])
                cats = sample.get('categories', [])

                if not bboxes:
                    continue

                file_name = f"{split_name}_{i:06d}"
                image_path = os.path.join(img_dir, f"{file_name}.jpg")
                image.save(image_path, "JPEG")

                label_path = os.path.join(lbl_dir, f"{file_name}.txt")
                save_yolo_label(bboxes, cats, img_w, img_h, label_path, mapping)

                pbar.update(1)

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        pbar.close()

    abs_root = os.path.abspath(output_dir)

    yaml_content = {
        'path': abs_root,
        'train': 'images/train',
        'val': 'images/val',
        'names': {k: v for k, v in enumerate(c_seg['class_names'])}
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logger.info(f"✅ Dataset preparation complete. Config saved to: {yaml_path}")

if __name__ == "__main__":
    prepare_doclaynet()