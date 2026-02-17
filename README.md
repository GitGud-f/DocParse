# DocPars: Intelligent Document Reconstruction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**DocPars** is a hybrid Computer Vision pipeline designed to transform high-variance smartphone document photos into structured, digital PDFs. It bridges the gap between raw pixel data and semantic document understanding using a combination of classical geometric processing and Deep Learning.

---

## Project Overview

Taking a photo of a document is easy; making it useful is hard. Smartphone photos suffer from perspective distortion, uneven lighting, shadows, and noise. This project implements a 4-phase pipeline to solving these issues:

1.  **Geometric Correction:** Rectifying the image to a top-down view.
2.  **Illumination Normalization:** removing shadows and enhancing contrast.
3.  **Layout Analysis (DL):** Understanding the structure (Headers, Tables, Figures).
4.  **Reconstruction:** Generating a searchable PDF that matches the original layout.

---

## 🛠️ Tech Stack

*   **Core:** Python, OpenCV, NumPy
*   **UI:** Streamlit (for interactive debugging and demo)
*   **Configuration:** YAML
*   **Evaluation:** Shapely (IoU), Levenshtein (OCR Accuracy)
*   **Planned (Phase II+):** PyTorch, YOLOv8

---

## 📂 Project Structure

```text
DocParse/
├── config.yaml             # Central configuration for thresholds & paths
├── app.py                  # Streamlit Dashboard (UI)
├── main.py                 # CLI Entry point for batch processing
├── src/
│   ├── scanner/
│   │   ├── geometry.py     # Canny, Contours, & Perspective Transform
│   │   ├── hough.py        # Hough Line Transform logic
│   │   └── filters.py      # Adaptive Thresholding & CLAHE
│   ├── evaluation/         # Metrics (IoU, CER, WER)
│   └── utils/              # Config loader & Image helpers
├── data/
│   ├── raw/                # Input images
│   ├── processed/          # Phase I outputs
│   └── benchmark/          # Golden dataset for evaluation
```

---

## 💻 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GitGud-f/DocParse.git
    cd DocParse
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

<!-- 3.  **Install Tesseract OCR Engine:**
    *   *Windows:* Download [installer](https://github.com/UB-Mannheim/tesseract/wiki). Add to PATH.
    *   *Linux:* `sudo apt install tesseract-ocr` -->

---

## Usage

### 1. Interactive UI (Recommended for Demo)
Launch the Streamlit dashboard to visualize every step of Phase I (Edge detection, Warping, Thresholding).
```bash
streamlit run app.py
```
*Features:* Switch between Blob/Hough methods, tune Canny thresholds in real-time, view Debug masks.

### 2. Batch Processing
Process all images in `data/raw/` and save results to `data/processed/`.
```bash
python main.py
```

### 3. Evaluation (Benchmarking)
Run the geometric evaluation against Ground Truth data.
```bash
python evaluate_phase1.py
```

---

## Current Progress (Phase I)

### Features Implemented
*   **Robust Corner Detection:**
    *   *Method A (Blob):* Hybrid edge detection (Gray + Saturation Channel) + Morphological closing + Contour approximation.
    *   *Method B (Hough):* Probabilistic Hough Transform + Line Clustering + Intersection calculation.
*   **Sub-pixel Refinement:** `cv2.cornerSubPix` implemented for high-precision homography.
*   **Perspective Transform:** 4-point transform to rectify document skew.
*   **Illumination Correction:**
    *   **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for texture preservation.
    *   **Adaptive Thresholding** for binarization (OCR prep).
*   **Evaluation Module:** Automated IoU (Geometric)

---

## Roadmap & TODOs

### Phase I: Preprocessing (Optimization)
- [x] Implement Basic Geometric Correction.
- [x] Implement Hybrid Edge Detection (Sat + Gray).
- [x] Create Evaluation Suite (IoU & CER).
- [ ] **TODO:** Implement "Document vs Table" detection logic to avoid cropping to internal tables.

### Phase II: Layout Analysis (Deep Learning)
- [ ] **TODO:** Select dataset (DocBank or PubLayNet).
- [ ] **TODO:** Generate synthetic training data using Phase I pipeline.
- [ ] **TODO:** Train YOLOv8-seg or Mask-RCNN to detect:
    - `Header`, `Footer`, `Paragraph`, `Image`, `Table`.
- [ ] **TODO:** Implement inference script in `src/segmentation/`.

### Phase III: OCR Integration
- [ ] **TODO:** Integrate Tesseract (or PaddleOCR) for text extraction.
- [ ] **TODO:** Implement logic to crop regions based on Phase II masks.
- [ ] **TODO:** Add Table Structure Recognition (Row/Col detection).

### Phase IV: Synthesis
- [ ] **TODO:** Create PDF generation module (`reportlab` or `fpdf`).
- [ ] **TODO:** Map OCR text to original coordinates in the PDF.

---

## Evaluation Results (Preliminary)

*   **Geometric Accuracy (IoU):** 89.4% (on internal test set of 20 images).
*   **OCR Legibility (CER):** TODO.

---

## Contributing
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---
