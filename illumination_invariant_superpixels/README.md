# Illumination Invariant Superpixels

**Authors:** Salman Awaise, Sameer Syed  
**Operating System Used:** Windows 11 (64-bit)

## Code Contributions

- **Salman Awaise**: Implemented the superpixel pipelines (`pipelines.py`), evaluation metrics (`metrics.py`), and experimental notebook (`notebooks/superpixel_processing.ipynb`), including stability, boundary IoU and VI analysis.

- **Sameer Syed**: Implemented the data loading and color constancy functions (`preprocessing.py`), SLIC segmentation and visualization (`superpixels.py`) and utility helpers (`utils.py`) used to manage images and label maps.

## How to Run This Project

### Project Overview

This project compares two pipelines for superpixel segmentation: raw RGB images and color-constancy corrected images. It uses the SLIC algorithm to generate superpixels under different lighting conditions and evaluates segmentation quality using Boundary Recall, Achievable Segmentation Accuracy (ASA), Stability, Variation of Information (VI), and Boundary IoU.

### Folder Structure

- `data/raw/`  
  Contains the input images used for all experiments.

- `src/`  
  Source code:
  - `pipelines.py` — raw pipeline and color-constancy pipeline  
  - `preprocessing.py` — image loading and color constancy  
  - `superpixels.py` — SLIC segmentation + boundary overlays  
  - `metrics.py` — superpixel evaluation metrics  
  - `utils.py` — saving/loading helpers  
  - `config.py` — directory paths and defaults  

- `results/`  
  Automatically generated outputs:
  - `figures/` — visualization grids  
  - `metrics/` — CSV files with metric summaries  

- `notebooks/superpixel_processing.ipynb`  
  Main notebook to run the full analysis.

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

### How to Run

**Step 1:** Open `notebooks/superpixel_processing.ipynb` in Jupyter Notebook or JupyterLab.

**Step 2:** Run the cells in order from top to bottom. The notebook will:
- Load images from `data/raw/`
- Apply raw and color-constancy pipelines
- Generate SLIC superpixels
- Compute evaluation metrics
- Save visualizations and CSV results

**Step 3:** View outputs:
- Visualizations → `results/figures/`
- Metrics summaries → `results/metrics/`

### Reproducibility Notes

You can adjust SLIC parameters inside the notebook:
- `n_segments` controls number of superpixels
- `compactness` controls clustering behavior
