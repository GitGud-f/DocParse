"""
DocParse Main Execution Script
Description: 
    Batch processes a folder of document images through the entire pipeline:
    Phase I (Geometry) -> Phase II (Layout) -> Phase III (OCR) -> Phase IV (Synthesis).
    Optionally merges all reconstructed single-page PDFs into one master PDF.
"""

import os
import cv2
import fitz  # PyMuPDF
import argparse
import logging
import shutil
from glob import glob

from src.utils.config import cfg
from src.scanner import geometry, filters
from src.segmentation.inference import LayoutAnalyzer
from src.ocr.engine import OCREngine
from src.synthesis.pdf_builder import PDFReconstructor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("DocParse-Batch")

def process_single_image(image_path, analyzer, ocr, pdf_builder, output_dir):
    """
    Passes a single image through the entire DocParse pipeline.
    """
    file_name = os.path.basename(image_path)
    base_name = os.path.splitext(file_name)[0]
    logger.info(f"🚀 Starting processing for: {file_name}")

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"❌ Failed to load image: {image_path}")
        return None

    # ==========================================
    # Phase I: Geometry & Illumination Preprocessing
    # ==========================================
    logger.info("   -> Phase I: Rectifying geometry & illumination...")
    corners = geometry.detect_document_corners(image)
    
    if corners is not None:
        warped = geometry.four_point_transform(image, corners)
    else:
        logger.warning("   -> ⚠️ Corner detection failed. Proceeding with original image.")
        warped = image.copy()

    # Deskew & Apply Filters
    warped = geometry.deskew_text_lines(warped)
    enhanced_gray = filters.clahe_equalization(warped)
    binarized = filters.adaptive_thresholding(enhanced_gray)

    # ==========================================
    # Phase II: Semantic Layout Segmentation
    # ==========================================
    logger.info("   -> Phase II: Semantic Layout Analysis (YOLO + XY-Cut)...")
    layout_elements = analyzer.predict(warped)
    
    if not layout_elements:
        logger.warning(f"   -> ⚠️ No layout elements detected in {file_name}. Skipping to next.")
        return None

    # ==========================================
    # Phase III: Hybrid OCR & Data Extraction
    # ==========================================
    logger.info("   -> Phase III: OCR & Table Parsing...")
    temp_crops_dir = os.path.join(output_dir, f"temp_{base_name}")
    os.makedirs(temp_crops_dir, exist_ok=True)

    ocr_results = ocr.process_layout(
        color_image=warped,
        binary_image=binarized,
        layout_elements=layout_elements,
        output_dir=temp_crops_dir
    )

    # ==========================================
    # Phase IV: PDF Synthesis
    # ==========================================
    logger.info("   -> Phase IV: Synthesizing Digital PDF...")
    pdf_filename = f"{base_name}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    pdf_builder.generate(
        original_image_shape=warped.shape,
        layout_data=ocr_results,
        output_path=pdf_path
    )
    
    # Cleanup temp crops to save disk space
    if os.path.exists(temp_crops_dir):
        shutil.rmtree(temp_crops_dir)

    logger.info(f"✅ Successfully generated: {pdf_path}")
    return pdf_path

def merge_pdfs(pdf_list, output_path):
    """
    Merges a list of PDF file paths into a single PDF document.
    """
    logger.info(f"📑 Merging {len(pdf_list)} PDFs into a single document...")
    merged_pdf = fitz.open()
    
    for pdf_file in pdf_list:
        try:
            with fitz.open(pdf_file) as doc:
                merged_pdf.insert_pdf(doc)
        except Exception as e:
            logger.error(f"❌ Failed to merge {pdf_file}: {e}")

    merged_pdf.save(output_path, garbage=4, deflate=True)
    merged_pdf.close()
    logger.info(f"🎉 Final merged PDF created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="DocParse Batch Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default=cfg['paths'].get('raw_data', 'data/raw'), 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default=cfg['paths'].get('output_data', 'data/output'), 
                        help="Directory to save generated PDFs")
    parser.add_argument("--merge", action="store_true", 
                        help="Pass this flag to merge all generated PDFs into a single file")
    parser.add_argument("--merged_name", type=str, default="Final_Document_Batch.pdf", 
                        help="Name of the final merged PDF file")
    
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find valid images
    supported_formats = ('*.jpg', '*.jpeg', '*.png')
    image_paths =[]
    for ext in supported_formats:
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))
        # Also check uppercase extensions (e.g. .JPG)
        image_paths.extend(glob(os.path.join(args.input_dir, ext.upper())))
        
    image_paths = sorted(image_paths)

    if not image_paths:
        logger.error(f"No images found in '{args.input_dir}'. Please add some images and try again.")
        return

    logger.info(f"Found {len(image_paths)} image(s) to process. Initializing Models...")

    # Load Deep Learning Models & Engines 
    # (Doing this once outside the loop saves huge amounts of memory/time)
    try:
        analyzer = LayoutAnalyzer()
        ocr_engine = OCREngine()
        pdf_builder = PDFReconstructor()
    except Exception as e:
        logger.error(f"Model initialization failed! Make sure your weights exist. Error: {e}")
        return

    generated_pdfs = []
    for img_path in image_paths:
        pdf_path = process_single_image(img_path, analyzer, ocr_engine, pdf_builder, args.output_dir)
        if pdf_path:
            generated_pdfs.append(pdf_path)
    logger.info(f"Batch processing complete. {len(generated_pdfs)} PDF(s) generated.")
    
    # Merge PDFs into one document depending on  --merge  flag
    if args.merge and generated_pdfs:
        path = './' + args.merged_name
        merge_pdfs(generated_pdfs, path)
        
if __name__ == "__main__":
    main()