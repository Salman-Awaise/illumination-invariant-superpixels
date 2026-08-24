# keeping the reflectance baseline experiment here
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# importing project configuration and preprocessing utilities
from . import config
from .preprocessing import load_image
# importing raw and color-constancy pipelines
from .pipelines import run_raw_pipeline, run_cc_pipeline
# importing SLIC for segmenting the ground-truth reflectance image
from .superpixels import run_slic
# importing the metrics used to score each pipeline against the reflectance
from .metrics import boundary_iou, variation_of_information

# setting the list of object names to evaluate
OBJECT_NAMES = ["apple", "deer", "cup1", "frog1"]


# loading the ground-truth reflectance image for an object from GT_DIR
def load_reflectance(object_name):
    # building the full path to the reflectance ground truth
    refl_path = os.path.join(config.GT_DIR, f"{object_name}_reflectance.png")
    # reading BGR image using OpenCV
    refl_bgr = cv2.imread(refl_path)
    # checking if image exists
    if refl_bgr is None:
        raise FileNotFoundError(f"reflectance ground truth not found: {refl_path}")
    # converting BGR to RGB for consistency
    refl_rgb = cv2.cvtColor(refl_bgr, cv2.COLOR_BGR2RGB)
    return refl_rgb


# comparing both pipelines against the reflectance baseline for every object
def run(object_names=OBJECT_NAMES):
    # creating a list to store all summary rows
    rows = []
    # running the reflectance comparison for each object
    for object_name in object_names:
        # listing all images for this object under varying illumination
        all_files = sorted(
            f for f in os.listdir(config.RAW_DIR)
            if f.startswith(object_name + "_") and f.endswith(".png"))
        # checking if images exist
        if len(all_files) == 0:
            print(f"[WARNING] no images for '{object_name}'")
            continue
        print(f"\n=== Processing object: {object_name} ===")
        print("Images:", all_files)
        # loading the illumination-free reflectance ground truth
        refl = load_reflectance(object_name)
        # segmenting the reflectance image to get the reference partition
        labels_refl = run_slic(refl)
        # creating lists for metrics across all lighting conditions
        iou_raw = []
        iou_cc = []
        vi_raw = []
        vi_cc = []
        # scoring each illumination condition against the reflectance partition
        for fname in all_files:
            print("  running pipelines on:", fname)
            # loading the image
            img = load_image(fname)
            # running raw SLIC pipeline
            labels_raw = run_raw_pipeline(img)
            # running color constancy SLIC pipeline
            img_cc, labels_cc = run_cc_pipeline(img, cc_method="gray_world")
            # computing boundary IoU against the reflectance partition
            iou_raw.append(boundary_iou(labels_raw, labels_refl))
            iou_cc.append(boundary_iou(labels_cc, labels_refl))
            # computing variation of information against the reflectance partition
            vi_raw.append(variation_of_information(labels_raw, labels_refl))
            vi_cc.append(variation_of_information(labels_cc, labels_refl))
        # converting lists to numpy arrays
        iou_raw = np.array(iou_raw)
        iou_cc = np.array(iou_cc)
        vi_raw = np.array(vi_raw)
        vi_cc = np.array(vi_cc)
        # storing summary metrics for this object
        rows.append({
            "object": object_name,
            "mean_iou_raw_vs_refl": float(iou_raw.mean()),
            "mean_iou_cc_vs_refl": float(iou_cc.mean()),
            "mean_vi_raw_vs_refl": float(vi_raw.mean()),
            "mean_vi_cc_vs_refl": float(vi_cc.mean()),
            "num_images": len(all_files),})
    # creating a dataframe with all object summaries
    df = pd.DataFrame(rows)
    print("\nReflectance baseline summary (boundary IoU + VI)")
    print(df)
    # saving the summary to CSV
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    out_csv = os.path.join(config.METRICS_DIR, "reflectance_baseline_summary.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved to:", out_csv)
    # plotting the comparison figures
    if len(df) > 0:
        plot_comparisons(df)
    return df


# plotting the reflectance baseline comparison figures
def plot_comparisons(df):
    # creating the figures directory if missing
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    x = np.arange(len(df))
    # plotting boundary IoU including the self-comparison sanity check
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.25, df["mean_iou_raw_vs_refl"], width=0.25, label="Raw vs Reflectance")
    plt.bar(x, df["mean_iou_cc_vs_refl"], width=0.25, label="CC vs Reflectance")
    plt.bar(x + 0.25, np.ones(len(df)), width=0.25, label="Reflectance vs Reflectance")
    plt.xticks(x, df["object"])
    plt.ylabel("Mean Boundary IoU")
    plt.title("Boundary IoU vs Reflectance Baseline")
    plt.legend()
    plt.tight_layout()
    _save(plt, "reflectance_iou_comparison.png")
    # plotting variation of information including the self-comparison sanity check
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.25, df["mean_vi_raw_vs_refl"], width=0.25, label="Raw vs Reflectance")
    plt.bar(x, df["mean_vi_cc_vs_refl"], width=0.25, label="CC vs Reflectance")
    plt.bar(x + 0.25, np.zeros(len(df)), width=0.25, label="Reflectance vs Reflectance")
    plt.xticks(x, df["object"])
    plt.ylabel("Mean Variation of Information")
    plt.title("VI vs Reflectance Baseline")
    plt.legend()
    plt.tight_layout()
    _save(plt, "reflectance_vi_comparison.png")
    # plotting boundary IoU for the two pipelines only
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, df["mean_iou_raw_vs_refl"], width=0.4, label="Raw vs Reflectance")
    plt.bar(x + 0.2, df["mean_iou_cc_vs_refl"], width=0.4, label="CC vs Reflectance")
    plt.xticks(x, df["object"])
    plt.ylim(0, 0.5)
    plt.ylabel("Mean Boundary IoU")
    plt.title("Boundary IoU vs Reflectance (Raw vs CC)")
    plt.legend()
    plt.tight_layout()
    _save(plt, "reflectance_iou_raw_vs_cc.png")
    # plotting variation of information for the two pipelines only
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, df["mean_vi_raw_vs_refl"], width=0.4, label="Raw vs Reflectance")
    plt.bar(x + 0.2, df["mean_vi_cc_vs_refl"], width=0.4, label="CC vs Reflectance")
    plt.xticks(x, df["object"])
    plt.ylabel("Mean Variation of Information")
    plt.title("VI vs Reflectance (Raw vs CC)")
    plt.legend()
    plt.tight_layout()
    _save(plt, "reflectance_vi_raw_vs_cc.png")


# saving the current figure into FIGURES_DIR
def _save(plt_module, filename):
    out_path = os.path.join(config.FIGURES_DIR, filename)
    plt_module.savefig(out_path, dpi=200)
    plt_module.show()
    print("Saved figure to:", out_path)
