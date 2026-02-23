import streamlit as st
import cv2
import numpy as np
import os
import imutils
import shutil
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from src.scanner import geometry, filters, hough
from src.utils.config import cfg

YOLO_AVAILABLE = False
OCR_AVAILABLE = False
PDF_AVAILABLE = False

try:
    from src.segmentation.inference import LayoutAnalyzer
    YOLO_AVAILABLE = True
except ImportError: pass

try:
    from src.ocr.engine import OCREngine
    OCR_AVAILABLE = True
except ImportError: pass

try:
    from src.synthesis.pdf_builder import PDFReconstructor
    PDF_AVAILABLE = True
except ImportError: pass

# --- Configuration & Styling ---
def setup_page():
    st.set_page_config(page_title="DocParse | HIAST", page_icon="📄", layout="wide", initial_sidebar_state="expanded")
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

# --- Helper Functions ---
def load_image_from_path(path): 
    return cv2.imread(path)

def load_image_from_upload(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def save_uploaded_file(uploaded_file):
    os.makedirs("data/raw", exist_ok=True)
    file_path = os.path.join("data/raw", uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def init_session_state():
    defaults = {
        'canvas_initial': None,
        'layout_elements': None,
        'ocr_results': None,
        'processed_image_hash': None,
        'canvas_img_hash': None,
        'canvas_objects': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# --- Model Caching ---
@st.cache_resource
def load_layout_model():
    if not YOLO_AVAILABLE: return None
    try:
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

# --- UI Components ---
def render_sidebar():
    st.sidebar.header("🔧 Settings")
    method = st.sidebar.radio("Detection Method:", ("Classical Contours (Blob)", "Hough Lines (Geometric)"))
    st.sidebar.divider()
    
    st.sidebar.markdown("**🛠 Developer Tools**")
    show_debug = st.sidebar.checkbox("Show Debug Edges/Mask", value=False, help="Visualize binary input.")
    
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
        else:
            st.sidebar.warning(f"No images found in {raw_dir}.")
            
    elif source_option == "Upload New Image":
        up = st.sidebar.file_uploader("Upload...", type=['jpg', 'jpeg', 'png'])
        if up:
            image = load_image_from_upload(up)
            if st.sidebar.button("Save to data/raw"): save_uploaded_file(up)
            
    return image, method, show_debug

def render_interactive_canvas(image_rgb, image_bgr, method, h_orig, w_orig):
    st.divider()
    st.markdown('<p class="header-style">3. Interactive Geometry Correction</p>', unsafe_allow_html=True)
    st.info("💡 Drag the 🔴 RED DOTS to the 4 corners of the document.")

    # Canvas Sizing
    CANVAS_WIDTH = 1000
    scale_factor = w_orig / CANVAS_WIDTH
    canvas_height = int(h_orig / scale_factor)
    
    bg_image_pil = Image.fromarray(cv2.resize(image_rgb, (CANVAS_WIDTH, canvas_height)))
    
    # Initialize Objects if New Image
    if st.session_state['canvas_objects'] is None:
        detected_corners = None
        try:
            if method == "Classical Contours (Blob)":
                detected_corners = geometry.detect_document_corners(image_bgr)
            else:
                detected_corners = hough.detect_hough_corners(image_bgr)
        except: pass
        
        # Default padding if detection fails
        if detected_corners is None:
            pad = 50
            detected_corners = np.array([
                [pad, pad], [w_orig-pad, pad], 
                [w_orig-pad, h_orig-pad], [pad, h_orig-pad]
            ], dtype="float32")
        
        ordered = geometry.order_points(detected_corners)
        scaled_pts = (ordered / scale_factor).tolist()
        
        objects = []
        for pt in scaled_pts:
            objects.append({
                "type": "circle", "left": pt[0], "top": pt[1],
                "originX": "center", "originY": "center",
                "fill": "rgba(255, 0, 0, 0.5)", "radius": 15,
                "stroke": "#fff", "strokeWidth": 2,
            })
        st.session_state['canvas_objects'] = {"version": "4.4.0", "objects": objects}

    # Render Canvas
    canvas_key = f"canvas_{st.session_state['processed_image_hash']}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        background_image=bg_image_pil,
        update_streamlit=True,
        height=canvas_height,
        width=CANVAS_WIDTH,
        drawing_mode="transform", 
        initial_drawing=st.session_state['canvas_objects'],
        key=canvas_key
    )
    
    # Calculate Final Corners
    final_corners = None
    
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        circles = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "circle"]
        if len(circles) == 4:
            pts = [[c["left"], c["top"]] for c in circles]
            final_corners = np.array(pts, dtype="float32") * scale_factor
            final_corners = geometry.order_points(final_corners)
            
    # Fallback to initial state
    if final_corners is None and st.session_state['canvas_objects']:
        pts = [[o['left'], o['top']] for o in st.session_state['canvas_objects']['objects']]
        final_corners = np.array(pts, dtype="float32") * scale_factor
        final_corners = geometry.order_points(final_corners)
        
    return final_corners

def process_image_pipeline(image, corners):
    # A. Warp
    warped = geometry.four_point_transform(image, corners)
    
    # B. Deskew
    with st.spinner("Refining Text Alignment..."):
        warped = geometry.deskew_text_lines(warped)
        
    # C. Filters
    with st.expander("Adjust Illumination Parameters"):
        c1, c2 = st.columns(2)
        clip = c1.slider("CLAHE Clip", 0.1, 5.0, 1.0)
        bs = c2.slider("Threshold Block Size", 3, 51, 29, step=2)

    enhanced_gray = filters.clahe_equalization(warped, clip_limit=clip, tile_grid_size=(8,8))
    binarized = filters.adaptive_thresholding(enhanced_gray, block_size=bs, c=15)
    
    # Display
    r1, r2, r3 = st.columns(3)
    r1.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="1. Rectified & Deskewed")
    r2.image(enhanced_gray, caption="2. Enhanced Grayscale")
    r3.image(binarized, caption="3. Binarized (For OCR)")
    
    return warped, binarized

# --- Main Application ---
def main():
    setup_page()
    init_session_state()

    st.title("📄 Intelligent Document Reconstruction")
    st.caption("Computer Vision Course Project - HIAST | Phases I & II")

    image, method, show_debug = render_sidebar()

    if image is None:
        st.info("Please select or upload an image to begin.")
        return

    # Handle Image Change Logic
    current_hash = hash(image.tobytes())
    if st.session_state['processed_image_hash'] != current_hash:
        st.session_state['processed_image_hash'] = current_hash
        st.session_state['canvas_img_hash'] = current_hash
        st.session_state['canvas_initial'] = None
        st.session_state['layout_elements'] = None
        st.session_state['canvas_objects'] = None

    # 1. Original & Debug View
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image.shape[:2]
    
    col1, col2 = st.columns([1, 1])
    col1.markdown('<p class="header-style">1. Original Input</p>', unsafe_allow_html=True)
    col1.image(image_rgb)
    
    col2.markdown(f'<p class="header-style">2. Debug View ({method})</p>', unsafe_allow_html=True)
    
    debug_view_image = None
    small_vis = imutils.resize(image, height=cfg['preprocessing']['resize_height'])
    
    if method == "Classical Contours (Blob)":
        debug_view_image = geometry.get_edges(small_vis, cfg['geometry']['blob_method'])
    elif method == "Hough Lines (Geometric)":
        debug_view_image = hough.get_processing_mask(small_vis)
        
    if show_debug and debug_view_image is not None:
        col2.image(debug_view_image, caption="Internal Edge/Mask Map")
    else:
        col2.info("Enable 'Show Debug Edges' in sidebar to see this.")

    # 2. Interactive Canvas
    final_corners = render_interactive_canvas(image_rgb, image, method, h_orig, w_orig)

    # 3. Processing Pipeline (Warp -> Filter)
    warped, binarized = None, None
    if final_corners is not None:
        warped, binarized = process_image_pipeline(image, final_corners)

    # 4. Phase II: Semantic Layout
    st.divider()
    st.markdown('<p class="header-style">4. Phase II: Semantic Layout Analysis</p>', unsafe_allow_html=True)

    if not YOLO_AVAILABLE:
        st.warning("Phase II modules not found.")
    elif warped is not None:
        if st.button("🧠 Run Deep Layout Analysis"):
            with st.spinner("Loading Model & Analyzing Layout..."):
                analyzer = load_layout_model()
                if analyzer:
                    st.session_state['layout_elements'] = analyzer.predict(warped)
                    
        if st.session_state['layout_elements']:
            elements = st.session_state['layout_elements']
            analyzer = load_layout_model()
            vis_layout = analyzer.visualize(warped, elements)
            
            l_col1, l_col2 = st.columns([2, 1])
            l_col1.image(cv2.cvtColor(vis_layout, cv2.COLOR_BGR2RGB), caption="Semantic Segmentation")
            
            with l_col2:
                df = pd.DataFrame([{
                    "ID": i, "Label": e['label'], "Conf": f"{e['confidence']:.2f}"
                } for i, e in enumerate(elements)])
                st.dataframe(df, height=300, hide_index=True)

    # 5. Phase III: Content Extraction
    st.divider()
    st.markdown('<p class="header-style">5. Phase III: Content Extraction (OCR)</p>', unsafe_allow_html=True)

    if not OCR_AVAILABLE:
        st.warning("OCR Engine not found.")
    elif st.session_state['layout_elements'] is None:
        st.info("⚠️ Please run Phase II first to detect regions.")
    else:
        if st.button("📝 Extract Text & Data (Phase III)"):
            with st.spinner("Running Hybrid OCR Extraction..."):
                ocr = load_ocr_engine()
                temp_output = os.path.join("data", "output", "streamlit_temp")
                if os.path.exists(temp_output): shutil.rmtree(temp_output)
                
                st.session_state['ocr_results'] = ocr.process_layout(
                    color_image=warped,
                    binary_image=binarized,
                    layout_elements=st.session_state['layout_elements'],
                    output_dir=temp_output
                )

        if st.session_state['ocr_results']:
            results = st.session_state['ocr_results']
            st.success(f"Successfully extracted {len(results)} elements.")

            for item in results:
                with st.container():
                    c_img, c_txt = st.columns([1, 3])
                    x1, y1, x2, y2 = item['bbox']
                    
                    if item['type'] == 'text':
                        display_crop = binarized[y1:y2, x1:x2]
                        caption = "Binary Input (OCR)"
                    else:
                        display_crop = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        caption = "Color Crop"
                        
                    c_img.image(display_crop, caption=caption, width=150)
                    with c_txt:
                        st.markdown(f"**ID {item['id']}: {item['label']}** *(Conf: {item['confidence']:.2f})*")
                        if item['type'] == 'text':
                            st.text_area("Extracted & Corrected Text:", value=item['content'], height=100, key=f"txt_{item['id']}")
                            
                        elif item['type'] == 'table':
                            st.info(f"🖼️ Image saved at: `{item['content']}`")
                            if item.get('table_data') and len(item['table_data']) > 1:
                                st.success("✅ Table Structure Parsed Successfully!")
                                
                                # --- SAFE DATAFRAME CREATION ---
                                raw_headers = item['table_data'][0]
                                safe_headers = []
                                # Make sure headers are unique and not empty
                                for c_idx, h in enumerate(raw_headers):
                                    h_clean = str(h).strip()
                                    if not h_clean:
                                        h_clean = f"Col_{c_idx+1}"
                                    # Handle duplicates (e.g., two columns named "Value")
                                    if h_clean in safe_headers:
                                        h_clean = f"{h_clean}_{c_idx+1}"
                                    safe_headers.append(h_clean)
                                
                                df = pd.DataFrame(item['table_data'][1:], columns=safe_headers)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.warning("⚠️ Borderless or complex table detected. Defaulting to hidden text fallback.")
                                if item.get('hidden_text'):
                                    st.text(item['hidden_text'])
                        else:
                            st.info(f"🖼️ Image saved at: `{item['content']}`")
                            if item.get('hidden_text'):
                                with st.expander("🔍 Show Hidden Searchable Text"):
                                    st.text(item['hidden_text'])
                    st.markdown("---")

    # 6. Phase IV: Synthesis
    st.divider()
    st.markdown('<p class="header-style">6. Phase IV: Final Reconstruction</p>', unsafe_allow_html=True)

    if not PDF_AVAILABLE:
        st.warning("PDF Builder module not found.")
    elif st.session_state['ocr_results'] is None:
        st.info("Run Phase III first to generate data for the PDF.")
    else:
        col_pdf1, col_pdf2 = st.columns([2, 1])
        col_pdf1.markdown("""This step maps the extracted content onto a new A4 canvas, 
            matching the original positions and font styles.""")
        
        with col_pdf2:
            if st.button("📄 Generate PDF"):
                with st.spinner("Synthesizing Document..."):
                    builder = PDFReconstructor()
                    pdf_filename = "reconstructed_doc.pdf"
                    pdf_path = os.path.join("data", "output", pdf_filename)
                    
                    builder.generate(
                        original_image_shape=warped.shape,
                        layout_data=st.session_state['ocr_results'],
                        output_path=pdf_path
                    )
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f.read(),
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                        st.success("PDF Created!")

if __name__ == "__main__":
    main()