import streamlit as st
import cv2
import numpy as np
import os
import imutils
import pandas as pd
from PIL import Image

# Import your custom modules
from src.scanner import geometry, filters, hough
from src.utils.config import cfg

# --- New Import for Phase II ---
# Wrap in try-except to prevent app crash if Phase II isn't fully set up yet
try:
    from src.segmentation.inference import LayoutAnalyzer
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

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

# --- Main ---
def main():
    st.title("📄 Intelligent Document Reconstruction")
    st.caption("Computer Vision Course Project - HIAST | Phases I & II")

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
                        elements = analyzer.predict(warped)
                            
                            # 2. Visualize
                        vis_layout = analyzer.visualize(warped, elements)
                            
                            # 3. Display
                        l_col1, l_col2 = st.columns([2, 1])
                            
                        with l_col1:
                            st.image(cv2.cvtColor(vis_layout, cv2.COLOR_BGR2RGB), 
                                    caption="YOLOv8 Segmentation Result")
                                         
                        with l_col2:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"**Detected Elements: {len(elements)}**")
                                
                                # Prepare data for table
                            table_data = []
                            for idx, el in enumerate(elements):
                                table_data.append({
                                        "ID": idx + 1,
                                        "Label": el['label'],
                                        "Conf": f"{el['confidence']:.2f}",
                                        "Position (Y)": int(el['bbox'][1]) # Sort visualizer
                                    })
                                
                            if table_data:
                                df = pd.DataFrame(table_data)
                                st.dataframe(df, hide_index=True)
                            else:
                                st.info("No layout elements detected.")
                                    
                            st.caption("Elements are sorted by reading order (Top-Down).")
                            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()