import streamlit as st
import cv2
import numpy as np
import os
import imutils
from PIL import Image

# Import your custom modules
from src.scanner import geometry, filters, hough
from src.utils.config import cfg

# --- Page Configuration ---
st.set_page_config(page_title="DocuStruct-CV | HIAST", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
<style>
    .main {background-color: #f5f5f5;} 
    .stButton>button {width: 100%; background-color: #4CAF50; color: white;} 
    .header-style {font-size:20px; font-weight:bold; font-family:sans-serif; color: #333;}
    .sub-header {font-size:16px; font-weight:bold; color: #666;}
    .debug-box {border: 2px dashed #ff4b4b; padding: 10px; border-radius: 5px; background-color: #fff0f0;}
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

# --- Main ---
def main():
    st.title("📄 Intelligent Document Reconstruction")
    st.caption("Computer Vision Course Project - HIAST | Phase I: Geometric & Illumination Correction")

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

    # --- Main Logic ---
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 1. Original & Detection ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<p class="header-style">1. Original Image</p>', unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.markdown(f'<p class="header-style">2. Corner Detection ({method})</p>', unsafe_allow_html=True)
            
            vis_image = image.copy()
            debug_view_image = None
            corners = None
            
            try:
                # Resize for consistent processing visualization
                # We use the config height for calculation, but for display we might want a specific size
                small_vis = imutils.resize(image, height=cfg['preprocessing']['resize_height'])

                if method == "Classical Contours (Blob)":
                    corners = geometry.detect_document_corners(image)
                    # Generate the edges specifically for debug visualization
                    debug_view_image = geometry.get_edges(small_vis, cfg['geometry']['blob_method'])
                    
                elif method == "Hough Lines (Geometric)":
                    corners = hough.detect_hough_corners(image)
                    # Generate the mask specifically for debug visualization
                    debug_view_image = hough.get_processing_mask(small_vis)

                # Draw Results on the visualization image
                if corners is not None:
                    box = geometry.order_points(corners).astype(int)
                    cv2.drawContours(vis_image, [box], -1, (0, 255, 0), 3)
                    for pt in box: cv2.circle(vis_image, tuple(pt), 10, (255, 0, 0), -1)
                    st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.error("No corners detected.")
            
            except Exception as e: st.error(f"Error: {e}")

        # --- Debug View Section (Conditional) ---
        if show_debug_edges and debug_view_image is not None:
            st.divider()
            st.markdown('<div class="debug-box">', unsafe_allow_html=True)
            st.markdown(f"**🔍 Debug View: {method} Internal Input**", unsafe_allow_html=True)
            
            d_col1, d_col2 = st.columns([3, 1])
            with d_col1:
                st.image(debug_view_image, caption="Binary Edge Map / Mask", use_container_width=True)
            with d_col2:
                st.info("""
                **What am I looking at?**
                
                This binary image is the actual input to the corner detection algorithms.
                
                - **White pixels:** Potential boundaries.
                - **Black pixels:** Background/Ignored.
                
                If the document edges are broken or missing here, detection will fail. Adjust `config.yaml` or lighting.
                """)
            st.markdown('</div>', unsafe_allow_html=True)


        # --- 2. Geometric & Illumination Results ---
        if corners is not None:
            st.divider()
            st.markdown('<p class="header-style">3. Pipeline Results</p>', unsafe_allow_html=True)
            
            # A. Geometric Transform
            warped = geometry.four_point_transform(image, corners)
            
            # B. Illumination Config
            with st.expander("Adjust Illumination Parameters"):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**CLAHE (Contrast)**")
                    clip = st.slider("Clip Limit", 0.1, 5.0, float(cfg['illumination']['clahe']['clip_limit']))
                    tile = cfg['illumination']['clahe']['tile_grid_size'][0] # Assume square
                with col_p2:
                    st.markdown("**Adaptive Threshold (Binarization)**")
                    bs = st.slider("Block Size", 3, 51, cfg['illumination']['adaptive_threshold']['block_size'], step=2)
                    c_val = st.slider("C Value", 1, 20, cfg['illumination']['adaptive_threshold']['c'])

            # C. Processing
            # Step 1: CLAHE (Gray)
            enhanced_gray = filters.clahe_equalization(
                warped, 
                clip_limit=clip, 
                tile_grid_size=(tile, tile)
            )
            
            # Step 2: Binarization (Black/White)
            binarized = filters.adaptive_thresholding(
                enhanced_gray, 
                block_size=bs, 
                c=c_val
            )

            # D. Display Three Columns
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.markdown('<p class="sub-header">A. Geometric Correction</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Top-Down Color", use_container_width=True)
                st.info("Input for future Color-based analysis.")

            with r2:
                st.markdown('<p class="sub-header">B. Enhanced Grayscale</p>', unsafe_allow_html=True)
                st.image(enhanced_gray, caption="CLAHE Output", use_container_width=True)
                st.success("**Input for Phase II (Deep Learning)**\n\nPreserves texture for Layout Analysis models (headers, figures, tables).")

            with r3:
                st.markdown('<p class="sub-header">C. Binarized Image</p>', unsafe_allow_html=True)
                st.image(binarized, caption="Adaptive Threshold", use_container_width=True)
                st.warning("**Input for Phase III (OCR)**\n\nHigh contrast text for Tesseract/PaddleOCR.")

if __name__ == "__main__":
    main()