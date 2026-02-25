# DocParse: Intelligent Document Reconstruction Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CV](https://img.shields.io/badge/Library-OpenCV-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Prototype-yellow)]()


**DocParse** is a hybrid computer vision pipeline designed to transform high-variance smartphone photos of documents into structured, searchable, and geometrically corrected PDF files. It bridges the gap between raw pixel data and digital document reconstruction using a combination of classical geometric computer vision and state-of-the-art Deep Learning models.

---

## 🚀 Key Features

### 📐 Phase I: Geometric Correction (Classical CV)
- **Automatic Corner Detection:** Utilizes multi-channel edge detection and morphological processing to find document boundaries.
- **Sub-pixel Refinement:** Refines corner coordinates to decimal precision for superior rectification.
- **Interactive Correction:** A Streamlit-based UI allows users to manually adjust corners if automatic detection fails.
- **Illumination Normalization:** Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and adaptive thresholding to remove shadows and uneven lighting.

### 🧠 Phase II: Semantic Layout Analysis (Deep Learning)
- **YOLO-DocLayNet Inference:** Utilize **YOLOv10** Fine-tuned on the **DocLayNet** dataset to segment the document into semantic regions:
  - `Title`, `Text`, `Table`, `Figure`, `Table Caption`, `Table Footer`, `Figure Caption`, `Formula`, `Formula Caption`.
- **Reading Order Resolution:** Implements the **Recursive XY-Cut algorithm** to sort detected elements into a natural reading order (top-down, left-right), handling multi-column layouts correctly.

### 📝 Phase III: Hybrid OCR & Content Extraction
- **Context-Aware OCR:** Dynamically adjusts Tesseract Page Segmentation Modes (PSM) based on the semantic label (e.g., treating a "Title" differently from a "Table Cell").
- **Table Structure Parsing:** Uses morphological projection profiles to reconstruct the grid structure of tables, converting them into editable data rather than static images.
- **Data Cleaning:** Includes heuristic post-processing and spellchecking to correct common OCR errors (e.g., `0` vs `O`).

### 📄 Phase IV: PDF Synthesis
- **Layout Preservation:** Reconstructs the document on a digital canvas using the detected coordinates.
- **Searchable Assets:** Embeds a hidden text layer behind images and tables, ensuring the entire PDF is searchable (Ctrl+F compatible).
- **Dynamic Typography:** Automatically scales font sizes to fit the text within the original bounding boxes.

---

## 🛠️ Project Structure

```text
DocParse/
├── data/
│   ├── raw/                  # Input images
│   └── output/               # Generated PDFs and debug visuals
├── models/
│   └── weights/              # YOLOv10 .pt checkpoints
├── src/
│   ├── scanner/              # Phase I: Geometry, Hough, Filters
│   ├── segmentation/         # Phase II: Inference, XY-Cut Sorting
│   ├── ocr/                  # Phase III: Tesseract Engine, Table Parser
│   ├── synthesis/            # Phase IV: PDF Generation (PyMuPDF)
│   └── utils/                # Config loader, image helpers
├── app.py                    # Streamlit Interactive Dashboard
├── config.yaml               # Centralized configuration
└── requirements.txt          # Python dependencies
```

---

## 💻 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/GitGud-f/DocParse.git
   cd DocParse
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   *   **Linux:** `sudo apt-get install tesseract-ocr`
   *   **Windows:** Download the installer from UB-Mannheim and set the path in `config.yaml`.

4. **Model Weights**
   Download the YOLO weights ([`doclayout_yolo_docstructbench_imgsz1024.pt`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt)) and place them in `models/weights/`.

---

## ⚡ Usage

### Running the Interactive Web App
The best way to experience the pipeline is via the Streamlit dashboard.

```bash
streamlit run app.py
```

1.  **Sidebar:** Select "Upload New Image" or choose a sample.
2.  **Phase I:** Verify the detected red corners. Drag them if necessary.
3.  **Phase II:** Click "Run Deep Layout Analysis" to see segmentation boxes.
4.  **Phase III:** Click "Extract Text & Data" to perform OCR.
5.  **Phase IV:** Click "Download PDF" to get the reconstructed document.

---

<!-- ## 📊 Pipeline Visualization

| 1. Input Image | 2. Geometric Correction | 3. Semantic Segmentation | 4. Final PDF |
| :---: | :---: | :---: | :---: |
| *(Raw Photo)* | *(Warped & Binarized)* | *(YOLO Detections)* | *(Reconstructed)* |
| ![Input](https://via.placeholder.com/150) | ![Warped](https://via.placeholder.com/150) | ![Seg](https://via.placeholder.com/150) | ![PDF](https://via.placeholder.com/150) |

--- -->

## ⚙️ Configuration
Modify `config.yaml` to tune parameters:

```yaml
preprocessing:
  resize_height: 1024

segmentation:
  model:
    conf_threshold: 0.45

ocr:
  lang: "eng"
  postprocessing:
    enable_spellcheck: True
```

---