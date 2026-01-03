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
st.markdown("""<style>.main {background-color: #f5f5f5;} .stButton>button {width: 100%; background-color: #4CAF50; color: white;} .header-style {font-size:20px; font-weight:bold; font-family:sans-serif;}</style>""", unsafe_allow_html=True)

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
    st.caption("Computer Vision Course Project - HIAST | Phase I: Geometric Correction")

    st.sidebar.header("🔧 Settings")
    method = st.sidebar.radio("Detection Method:", ("Classical Contours (Blob)", "Hough Lines (Geometric)"))
    
    st.sidebar.header("📂 Input Source")
    source_option = st.sidebar.radio("Select Image Source:", ("Select from data/raw", "Upload New Image"))

    image = None
    if source_option == "Select from data/raw":
        raw_dir = "data/raw"
        os.makedirs(raw_dir, exist_ok=True)
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            selected = st.sidebar.selectbox("Choose an image:", files)
            if selected: image = load_image_from_path(os.path.join(raw_dir, selected))
        else: st.sidebar.warning("No images found in data/raw.")
    elif source_option == "Upload New Image":
        up = st.sidebar.file_uploader("Upload...", type=['jpg', 'jpeg', 'png'])
        if up:
            image = load_image_from_upload(up)
            if st.sidebar.button("Save to data/raw"): save_uploaded_file(up)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<p class="header-style">1. Original Image</p>', unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.markdown(f'<p class="header-style">2. Corner Detection ({method})</p>', unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Detected Corners", "Algorithm View"])
            
            # --- VISUALIZATION LOGIC ---
            vis_image = image.copy()
            debug_mask = None
            corners = None
            
            try:
                # Resize for consistency in debug visualization
                small_vis = imutils.resize(image, height=600)

                if method == "Classical Contours (Blob)":
                    # 1. Run Detection
                    corners = geometry.detect_document_corners(image)
                    
                    # 2. Re-create the mask for display (Logic copied from geometry.py)
                    gray = cv2.cvtColor(small_vis, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edged = cv2.Canny(blurred, 75, 200)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
                    debug_mask = cv2.dilate(closed, kernel, iterations=1) # The "Blob"
                    
                elif method == "Hough Lines (Geometric)":
                    # 1. Run Detection
                    corners = hough.detect_hough_corners(image)
                    
                    # 2. Get the mask actually used by the algorithm
                    debug_mask = hough.get_processing_mask(small_vis)

                # --- DRAW RESULTS ---
                if corners is not None:
                    # Draw Green Box
                    box = geometry.order_points(corners).astype(int)
                    cv2.drawContours(vis_image, [box], -1, (0, 255, 0), 3)
                    for pt in box: cv2.circle(vis_image, tuple(pt), 10, (255, 0, 0), -1)
                    
                    with tab1: st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    with tab1: st.error("No corners detected.")

                with tab2:
                    if debug_mask is not None:
                        st.image(debug_mask, caption=f"Internal Mask ({method})", clamp=True, use_container_width=True)
                        st.caption("This binary image is what the algorithm analyzes.")
            
            except Exception as e: st.error(f"Error: {e}")

        if corners is not None:
            st.divider()
            st.markdown('<p class="header-style">3. Result</p>', unsafe_allow_html=True)
            
            # Use the filters.preprocess_image wrapper (Logic check)
            # Note: We pass the raw image and corners manually here for granular control in UI
            warped = geometry.four_point_transform(image, corners)
            
            # Illumination
            c_params = st.columns([1, 2])
            with c_params[0]:
                def_bs = cfg['illumination']['adaptive_threshold']['block_size']
                def_c = cfg['illumination']['adaptive_threshold']['c']
                
                bs = st.slider("Block Size", 3, 51, def_bs, step=2)
                c_val = st.slider("C Value", 1, 20, def_c)
            
            enhanced_gray = filters.clahe_equalization(warped)
            binarized = filters.adaptive_thresholding(enhanced_gray, block_size=bs, c=c_val)
            
            with c_params[1]:
                st.image(binarized, caption="Final Preprocessed Document", use_container_width=True)

if __name__ == "__main__":
    main()