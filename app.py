import streamlit as st
import cv2
import numpy as np
import os
import imutils
import shutil
import pandas as pd
from PIL import Image

# Import your custom modules
from src.scanner import geometry, filters, hough
from src.utils.config import cfg

# --- New Import for Phase II ---
# Wrap in try-except to prevent app crash if Phase II isn't fully set up yet
PHASE2_AVAILABLE = False
OCR_AVAILABLE = False
PDF_AVAILABLE = False

try:
    from src.segmentation.inference import LayoutAnalyzer
    PHASE2_AVAILABLE = True
except ImportError:
    pass

try:
    from src.ocr.engine import OCREngine
    OCR_AVAILABLE = True
except ImportError:
    pass

try:
    from src.synthesis.pdf_builder import PDFReconstructor
    PDF_AVAILABLE = True
except ImportError:
    pass

# --- Page Configuration ---
st.set_page_config(page_title="DocParse | HIAST", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
<style>
    .main {background-color: #f5f5f5;} 
    .stButton>button {width: 100%; background-color: #4CAF50; color: white;} 
    .header-style {font-size:20px; font-weight:bold; font-family:sans-serif; color: #333;}
    .sub-header {font-size:16px; font-weight:bold; color: #666;}
    .debug-box {border: 2px dashed #ff4b4b; padding: 10px; border-radius: 5px; background-color: #fff0f0;}
    .success-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f0fff0;}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def load_image_from_path(path): return cv2.imread(path)
def load_image_from_upload(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)
def save_uploaded_file(uploaded_file):
    os.makedirs("data/raw", exist_ok=True)
    file_path = os.path.join("data/raw", uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

# --- Model Caching (Phase II) ---
@st.cache_resource
def load_layout_model():
    if not PHASE2_AVAILABLE: return None
    try:
        # LayoutAnalyzer automatically looks in config for weights path
        return LayoutAnalyzer() 
    except Exception as e:
        st.error(f"Failed to load Phase II Model: {e}")
        return None
    
@st.cache_resource
def load_ocr_engine():
    if not OCR_AVAILABLE: return None
    try:
        return OCREngine()
    except Exception as e:
        st.error(f"Failed to load OCR Engine: {e}")
        return None

# --- Main ---
def main():
    st.title("📄 Intelligent Document Reconstruction")
    st.caption("Computer Vision Course Project - HIAST | Phases I & II")

    # --- Session State Management ---
    if 'layout_elements' not in st.session_state:
        st.session_state['layout_elements'] = None
    if 'ocr_results' not in st.session_state:
        st.session_state['ocr_results'] = None
    if 'processed_image_hash' not in st.session_state:
        st.session_state['processed_image_hash'] = None
        
    # --- Sidebar ---
    st.sidebar.header("🔧 Settings")
    method = st.sidebar.radio("Detection Method:", ("Classical Contours (Blob)", "Hough Lines (Geometric)"))
    
    st.sidebar.divider()
    
    # --- Debug Toggle ---
    st.sidebar.markdown("**🛠 Developer Tools**")
    show_debug_edges = st.sidebar.checkbox("Show Debug Edges/Mask", value=False, help="Visualize the binary input used for detection.")

    st.sidebar.header("📂 Input Source")
    source_option = st.sidebar.radio("Select Image Source:", ("Select from data/raw", "Upload New Image"))

    image = None
    if source_option == "Select from data/raw":
        raw_dir = cfg['paths']['raw_data']
        os.makedirs(raw_dir, exist_ok=True)
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            selected = st.sidebar.selectbox("Choose an image:", files)
            if selected: image = load_image_from_path(os.path.join(raw_dir, selected))
        else: st.sidebar.warning(f"No images found in {raw_dir}.")
    elif source_option == "Upload New Image":
        up = st.sidebar.file_uploader("Upload...", type=['jpg', 'jpeg', 'png'])
        if up:
            image = load_image_from_upload(up)
            if st.sidebar.button("Save to data/raw"): save_uploaded_file(up)

    if image is not None:
        current_hash = hash(image.tobytes())
        if st.session_state['processed_image_hash'] != current_hash:
            st.session_state['layout_elements'] = None
            st.session_state['ocr_results'] = None
            st.session_state['processed_image_hash'] = current_hash
    # --- Main Logic ---
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 1. Original & Detection ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<p class="header-style">1. Original Image</p>', unsafe_allow_html=True)
            st.image(image_rgb)

        with col2:
            st.markdown(f'<p class="header-style">2. Corner Detection ({method})</p>', unsafe_allow_html=True)
            
            vis_image = image.copy()
            debug_view_image = None
            corners = None
            
            try:
                # Resize for consistent processing visualization
                small_vis = imutils.resize(image, height=cfg['preprocessing']['resize_height'])

                if method == "Classical Contours (Blob)":
                    corners = geometry.detect_document_corners(image)
                    debug_view_image = geometry.get_edges(small_vis, cfg['geometry']['blob_method'])
                    
                elif method == "Hough Lines (Geometric)":
                    corners = hough.detect_hough_corners(image)
                    debug_view_image = hough.get_processing_mask(small_vis)

                # Draw Results
                if corners is not None:
                    box = geometry.order_points(corners).astype(int)
                    cv2.drawContours(vis_image, [box], -1, (0, 255, 0), 3)
                    for pt in box: cv2.circle(vis_image, tuple(pt), 10, (255, 0, 0), -1)
                    st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                else:
                    st.error("No corners detected.")
            
            except Exception as e: st.error(f"Error: {e}")

        # --- Debug View Section ---
        if show_debug_edges and debug_view_image is not None:
            st.divider()
            st.markdown('<div class="debug-box">', unsafe_allow_html=True)
            st.markdown(f"**🔍 Debug View: {method} Internal Input**", unsafe_allow_html=True)
            d_col1, d_col2 = st.columns([3, 1])
            with d_col1:
                st.image(debug_view_image, caption="Binary Edge Map / Mask")
            with d_col2:
                st.info("White pixels: Potential boundaries.\nBlack pixels: Background.")
            st.markdown('</div>', unsafe_allow_html=True)


        # --- 3. Pipeline Results (Phase I) ---
        # --- MODIFIED LOGIC START ---
        if corners is not None:
            # SUCCESS CASE
            box = geometry.order_points(corners).astype(int)
            cv2.drawContours(vis_image, [box], -1, (0, 255, 0), 3)
            for pt in box: cv2.circle(vis_image, tuple(pt), 10, (255, 0, 0), -1)
            st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), caption="Detected Document")
                
            # Create the warped image for the next step
            warped = geometry.four_point_transform(image, corners)
                
        else:
            # FAILURE / FALLBACK CASE
            st.warning("⚠️ Corners not detected.")
            st.info("Fallback: Using the full original image for the next steps.")
                
            # Visual feedback that we are using the whole image
            h, w = image.shape[:2]
            cv2.rectangle(vis_image, (5, 5), (w-5, h-5), (255, 0, 0), 5)
            st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), caption="Fallback: Full Image Used")
                
            # Important: Set 'warped' to original so downstream code doesn't crash
            warped = image.copy() 
            # --- MODIFIED LOGIC END ---
        st.divider()
        st.markdown('<p class="header-style">3. Phase I: Geometric & Illumination</p>', unsafe_allow_html=True)
            
            
            # B. Illumination Config
        with st.expander("Adjust Illumination Parameters"):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**CLAHE (Contrast)**")
                clip = st.slider("Clip Limit", 0.1, 5.0, float(cfg['illumination']['clahe']['clip_limit']))
                tile = cfg['illumination']['clahe']['tile_grid_size'][0]
            with col_p2:
                st.markdown("**Adaptive Threshold (Binarization)**")
                bs = st.slider("Block Size", 3, 51, cfg['illumination']['adaptive_threshold']['block_size'], step=2)
                c_val = st.slider("C Value", 1, 20, cfg['illumination']['adaptive_threshold']['c'])

            # C. Processing
        enhanced_gray = filters.clahe_equalization(warped, clip_limit=clip, tile_grid_size=(tile, tile))
        binarized = filters.adaptive_thresholding(enhanced_gray, block_size=bs, c=c_val)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('<p class="sub-header">A. Geometric Correction</p>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Top-Down Color")
        with r2:
            st.markdown('<p class="sub-header">B. Enhanced Grayscale</p>', unsafe_allow_html=True)
            st.image(enhanced_gray, caption="CLAHE Output")
        with r3:
            st.markdown('<p class="sub-header">C. Binarized Image</p>', unsafe_allow_html=True)
            st.image(binarized, caption="Adaptive Threshold")

            # --- 4. Phase II: Semantic Layout ---
        st.divider()
        st.markdown('<p class="header-style">4. Phase II: Semantic Layout Analysis</p>', unsafe_allow_html=True)

        if not PHASE2_AVAILABLE:
            st.warning("Phase II modules not found. Please implement src/segmentation/inference.py")
        else:
            # We use the 'warped' image (Color) for YOLO inference
            if st.button("🧠 Run Deep Layout Analysis"):
                with st.spinner("Loading Model & Analyzing Layout..."):
                    analyzer = load_layout_model()
                        
                    if analyzer:
                            # 1. Predict
                        st.session_state['layout_elements'] = analyzer.predict(warped)
                        st.session_state['layout_elements']
                        
            if st.session_state['layout_elements']:
                elements = st.session_state['layout_elements']
                analyzer = load_layout_model()
                vis_layout = analyzer.visualize(warped, elements)
                            
                            # 3. Display
                l_col1, l_col2 = st.columns([2, 1])
                l_col1.image(cv2.cvtColor(vis_layout, cv2.COLOR_BGR2RGB), caption="Semantic Segmentation")
                            
                with l_col2:
                    df = pd.DataFrame([{
                        "ID": i, "Label": e['label'], "Conf": f"{e['confidence']:.2f}"
                    } for i, e in enumerate(elements)])
                    st.dataframe(df, height=300, hide_index=True)

        st.divider()
        st.markdown('<p class="header-style">5. Phase III: Content Extraction (OCR)</p>', unsafe_allow_html=True)

        if not OCR_AVAILABLE:
            st.warning("OCR Engine (Tesseract) not found or not configured.")
        elif st.session_state['layout_elements'] is None:
            st.info("⚠️ Please run Phase II first to detect regions.")
        else:
            if st.button("📝 Extract Text & Data (Phase III)"):
                with st.spinner("Running Hybrid OCR Extraction..."):
                    ocr = load_ocr_engine()
                    
                    # Create temp output for crops
                    temp_output = os.path.join("data", "output", "streamlit_temp")
                    if os.path.exists(temp_output): shutil.rmtree(temp_output)
                    
                    # Run Pipeline
                    st.session_state['ocr_results'] = ocr.process_layout(
                        color_image=warped,           # For Images/Tables
                        binary_image=binarized,       # For Text (Better Accuracy)
                        layout_elements=st.session_state['layout_elements'],
                        output_dir=temp_output
                    )

            # Display Results
            if st.session_state['ocr_results']:
                results = st.session_state['ocr_results']
                st.success(f"Successfully extracted {len(results)} elements.")

                for item in results:
                    with st.container():
                        c_img, c_txt = st.columns([1, 3])
                        
                        # Left: Visual Crop
                        x1, y1, x2, y2 = item['bbox']
                        # Crop on the fly for display to ensure it matches
                        if item['type'] == 'text':
                            # Show binary crop for text (what the OCR saw)
                            display_crop = binarized[y1:y2, x1:x2]
                            caption = "Binary Input (OCR)"
                        else:
                            # Show color crop for visuals
                            display_crop = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                            caption = "Color Crop"
                            
                        c_img.image(display_crop, caption=caption, width=150)
                        
                        # Right: Content
                        with c_txt:
                            st.markdown(f"**ID {item['id']}: {item['label']}** ({item['confidence']:.2f})")
                            if item['type'] == 'text':
                                st.text_area("Extracted Text:", value=item['content'], height=100, key=f"txt_{item['id']}")
                            else:
                                st.info(f"🖼️ Image saved at: `{item['content']}`")
                        
                        st.markdown("---")
                        
        # --- 6. Phase IV: Synthesis ---
    st.divider()
    st.markdown('<p class="header-style">6. Phase IV: Final Reconstruction</p>', unsafe_allow_html=True)
    


    if not PDF_AVAILABLE:
        st.warning("PDF Builder module not found.")
    elif st.session_state['ocr_results'] is None:
        st.info("Run Phase III first to generate data for the PDF.")
    else:
        col_pdf1, col_pdf2 = st.columns([2, 1])
        
        with col_pdf1:
            st.markdown("""
            This step maps the extracted content onto a new A4 canvas, 
            matching the original positions and font styles.
            """)
        
        with col_pdf2:
            if st.button("📄 Generate PDF"):
                with st.spinner("Synthesizing Document..."):
                    builder = PDFReconstructor()
                    
                    # Define Output Path
                    pdf_filename = "reconstructed_doc.pdf"
                    pdf_path = os.path.join("data", "output", pdf_filename)
                    
                    # Generate
                    builder.generate(
                        original_image_shape=warped.shape,
                        layout_data=st.session_state['ocr_results'],
                        output_path=pdf_path
                    )
                    
                    # Read file for download button
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        
                    st.success("PDF Created!")
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
if __name__ == "__main__":
    main()