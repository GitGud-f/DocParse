import streamlit as st
import cv2
import numpy as np
import os
import imutils
from PIL import Image

# Import your custom modules
from src.scanner import geometry, filters

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuStruct-CV | HIAST",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .header-style {
        font-size:20px;
        font-weight:bold;
        font-family:sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_image_from_path(path):
    return cv2.imread(path)

def load_image_from_upload(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image

def save_uploaded_file(uploaded_file):
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Main App ---
def main():
    st.title("📄 Intelligent Document Reconstruction")
    st.caption("Computer Vision Course Project - HIAST | Phase I: Geometric Correction & Preprocessing")

    # --- Sidebar: Input Selection ---
    st.sidebar.header("📂 Input Source")
    source_option = st.sidebar.radio("Select Image Source:", ("Select from data/raw", "Upload New Image"))

    image = None
    filename = ""

    if source_option == "Select from data/raw":
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            st.sidebar.warning("No images found in data/raw.")
        else:
            selected_file = st.sidebar.selectbox("Choose an image:", files)
            if selected_file:
                image_path = os.path.join(raw_dir, selected_file)
                image = load_image_from_path(image_path)
                filename = selected_file

    elif source_option == "Upload New Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = load_image_from_upload(uploaded_file)
            filename = uploaded_file.name
            # Save it so we can use it later
            if st.sidebar.button("Save to data/raw"):
                save_path = save_uploaded_file(uploaded_file)
                st.sidebar.success(f"Saved to {save_path}")

    # --- Main Processing Loop ---
    if image is not None:
        # Convert BGR to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Row 1: Original Image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<p class="header-style">1. Original Image</p>', unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)
            st.info(f"Dimensions: {image.shape[1]}x{image.shape[0]}")

        # --- Step 2: Corner Detection ---
        with col2:
            st.markdown('<p class="header-style">2. Corner Detection (The "Blob" Method)</p>', unsafe_allow_html=True)
            
            # Create a debug visualization for the UI
            debug_tab1, debug_tab2 = st.tabs(["Detected Corners", "Morphology View"])
            
            # Run the detection
            try:
                # We re-implement a bit of logic here just to visualize the Morph steps
                # for the "Morphology View" tab
                gray = cv2.cvtColor(imutils.resize(image, height=800), cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, 75, 200)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
                dilated = cv2.dilate(closed, kernel, iterations=1)
                
                # Run the actual function
                corners = geometry.detect_document_corners(image)
                
                if corners is not None:
                    # Draw corners on copy
                    vis_image = image.copy()
                    cv2.drawContours(vis_image, [corners.astype(int)], -1, (0, 255, 0), 5)
                    for pt in corners:
                        cv2.circle(vis_image, tuple(pt.astype(int)), 10, (255, 0, 0), -1)
                    
                    with debug_tab1:
                        st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with debug_tab2:
                        st.image(dilated, caption="Morphology (Dilated Edges)", clamp=True, use_container_width=True)
                        st.caption("This binary blob is what the algorithm actually 'sees' to find the paper.")
                else:
                    st.error("Could not detect corners! Try adjusting the background or lighting.")
                    corners = None
            except Exception as e:
                st.error(f"Error in detection: {e}")
                corners = None

        if corners is not None:
            st.divider()
            
            # --- Step 3: Perspective Transform ---
            st.markdown('<p class="header-style">3. Geometric Correction (Unwarping)</p>', unsafe_allow_html=True)
            
            warped = geometry.four_point_transform(image, corners)
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(warped_rgb, caption="Top-Down View", use_container_width=True)
            
            with c2:
                # Calculate Aspect Ratio
                h, w = warped.shape[:2]
                ar = h / w if h > w else w / h
                st.metric(label="Calculated Aspect Ratio", value=f"{ar:.2f}", delta=f"{ar - 1.414:.2f} off A4")
                if 1.35 < ar < 1.48:
                    st.success("✅ Aspect Ratio matches A4 paper closely.")
                else:
                    st.warning("⚠️ Aspect Ratio is unusual. Perspective might be off.")

            st.divider()

            # --- Step 4: Illumination & Binarization ---
            st.markdown('<p class="header-style">4. Illumination Normalization & OCR Prep</p>', unsafe_allow_html=True)

            col_params, col_result = st.columns([1, 2])
            
            with col_params:
                st.markdown("**Filters Parameters**")
                apply_clahe = st.checkbox("Apply CLAHE (Contrast)", value=True)
                
                block_size = st.slider("Block Size (odd number)", 3, 51, 15, step=2)
                c_value = st.slider("C Constant (subtraction)", 1, 20, 5)
                
                # Apply Filters
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                
                if apply_clahe:
                    processed = filters.clahe_equalization(warped_gray)
                else:
                    processed = warped_gray
                
                binarized = filters.adaptive_thresholding(processed, block_size=block_size, c=c_value)

            with col_result:
                tab_gray, tab_bin = st.tabs(["Enhanced Grayscale", "Final Binarized (OCR Ready)"])
                with tab_gray:
                    st.image(processed, use_container_width=True, clamp=True)
                with tab_bin:
                    st.image(binarized, use_container_width=True, clamp=True)
                    
            # --- Save Button ---
            if st.button("Download Processed PDF/Image"):
                st.info("PDF Generation coming in Phase IV! For now, image saved to data/processed.")
                # Save logic here if needed
                
    else:
        st.info("👈 Please select an image from the sidebar to begin.")

if __name__ == "__main__":
    main()