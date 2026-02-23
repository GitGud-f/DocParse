"""
Module: train

Description: 
    Handles the training loop for the YOLOv8 Semantic Layout Segmentation model.
    It loads config, executes training, and saves the best model to the weights directory.
    
Functions:
    - train_layout_model: Main function to execute the training pipeline.
"""

import os
import shutil
import logging
from ultralytics import YOLO
from src.utils.config import cfg

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_layout_model():
    """
    Executes the YOLOv8 training pipeline using parameters from config.yaml.
    """
    
    c_seg = cfg['segmentation']
    c_model = c_seg['model']
    c_data = c_seg['dataset']
    
    dataset_yaml_path = os.path.abspath(os.path.join(c_data['training_dir'], "dataset.yaml"))
    
    if not os.path.exists(dataset_yaml_path):
        logger.error(f"Dataset config not found at: {dataset_yaml_path}")
        logger.error("Please run 'python -m src.segmentation.dataset_utils' first.")
        return

    logger.info(f"--- Starting Training Pipeline ---")
    logger.info(f"Model: {c_model['name']}")
    logger.info(f"Epochs: {c_model['epochs']}")
    logger.info(f"Data Config: {dataset_yaml_path}")

    try:
        model = YOLO(c_model['name'])
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
        return

    project_dir = os.path.abspath("models/runs")
    experiment_name = "layout_segmentation_v1"

    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=c_model['epochs'],
            imgsz=c_model['img_size'],
            batch=c_model['batch_size'],
            patience=10,            # Early stopping if no improvement for 10 epochs
            project=project_dir,    
            name=experiment_name,   
            exist_ok=True,          
            verbose=True,
            device=0,               # Use GPU 0. Change to 'cpu' if no GPU available.
            plots=True
        )
        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training interrupted or failed: {e}")
        return
    
    source_weights = os.path.join(project_dir, experiment_name, "weights", "best.pt")
    dest_dir = os.path.join("models", "weights")
    dest_weights = os.path.join(dest_dir, "best_layout.pt")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(source_weights):
        shutil.copy2(source_weights, dest_weights)
        logger.info(f"🏆 Best model weights saved to: {dest_weights}")
        logger.info("You can now update 'config.yaml' to point to this path for inference.")
    else:
        logger.warning(f"Could not find generated weights at {source_weights}")

if __name__ == "__main__":
    train_layout_model()