# keeping the stability analysis experiment here
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# importing project configuration and preprocessing utilities
from . import config
from .preprocessing import load_image
# importing raw and color-constancy pipelines
from .pipelines import run_raw_pipeline, run_cc_pipeline
# importing stability metrics for comparison across lighting conditions
from .metrics import compute_stability, boundary_iou, variation_of_information

# setting the list of object names to evaluate
OBJECT_NAMES = ["apple", "deer", "cup1", "frog1"]


# running the stability analysis across all objects
def run(object_names=OBJECT_NAMES):
    # creating a list to store all summary rows
    rows = []
    # running stability analysis for each object
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
        # selecting the first image as the reference
        ref_name = all_files[0]
        print("Reference image:", ref_name)
        # creating dictionaries to store superpixel labels
        labels_raw_dict = {}
        labels_cc_dict = {}
        # running raw and color constancy pipelines on all images
        for fname in all_files:
            print("  running pipelines on:", fname)
            # loading the image
            img = load_image(fname)
            # running raw SLIC pipeline
            labels_raw = run_raw_pipeline(img)
            # running color constancy SLIC pipeline
            img_cc, labels_cc = run_cc_pipeline(img, cc_method="gray_world")
            # storing computed labels
            labels_raw_dict[fname] = labels_raw
            labels_cc_dict[fname] = labels_cc
        # getting reference segmentations
        ref_raw = labels_raw_dict[ref_name]
        ref_cc = labels_cc_dict[ref_name]
        # creating lists for metrics across all lighting conditions
        stab_raw = []
        stab_cc = []
        b_iou_raw = []
        b_iou_cc = []
        vi_raw = []
        vi_cc = []
        # computing metrics for each illumination condition
        for fname in all_files:
            lr = labels_raw_dict[fname]
            lc = labels_cc_dict[fname]
            # computing neighbor-based stability
            stab_raw.append(compute_stability(ref_raw, lr))
            stab_cc.append(compute_stability(ref_cc, lc))
            # computing boundary IoU
            b_iou_raw.append(boundary_iou(ref_raw, lr))
            b_iou_cc.append(boundary_iou(ref_cc, lc))
            # computing variation of information
            vi_raw.append(variation_of_information(ref_raw, lr))
            vi_cc.append(variation_of_information(ref_cc, lc))
        # converting lists to numpy arrays
        stab_raw = np.array(stab_raw)
        stab_cc = np.array(stab_cc)
        b_iou_raw = np.array(b_iou_raw)
        b_iou_cc = np.array(b_iou_cc)
        vi_raw = np.array(vi_raw)
        vi_cc = np.array(vi_cc)
        # storing summary metrics for this object
        rows.append({
            "object": object_name,
            "num_images": len(all_files),
            "mean_neighbor_stab_raw": float(stab_raw.mean()),
            "mean_neighbor_stab_cc": float(stab_cc.mean()),
            "mean_boundary_iou_raw": float(b_iou_raw.mean()),
            "mean_boundary_iou_cc": float(b_iou_cc.mean()),
            "mean_vi_raw": float(vi_raw.mean()),
            "mean_vi_cc": float(vi_cc.mean()),})
    # creating a dataframe with all object summaries
    df = pd.DataFrame(rows)
    print("\nStability summary (neighbor + boundary IoU + VI)")
    print(df)
    # saving the summary to CSV
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    out_csv = os.path.join(config.METRICS_DIR, "stability_summary_all_metrics.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved to:", out_csv)
    # plotting boundary IoU comparison between pipelines
    if len(df) > 0:
        x = np.arange(len(df))
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.15, df["mean_boundary_iou_raw"], width=0.3, label="Raw")
        plt.bar(x + 0.15, df["mean_boundary_iou_cc"], width=0.3, label="CC (gray-world)")
        plt.xticks(x, df["object"])
        plt.ylabel("Mean boundary IoU")
        plt.title("Boundary-based superpixel stability (Raw vs CC)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plotting variation of information comparison between pipelines
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.15, df["mean_vi_raw"], width=0.3, label="Raw")
        plt.bar(x + 0.15, df["mean_vi_cc"], width=0.3, label="CC (gray-world)")
        plt.xticks(x, df["object"])
        plt.ylabel("Mean VI (lower = more similar)")
        plt.title("Variation of Information vs reference (Raw vs CC)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return df
