# keeping the illumination grid visualization here
import os
import matplotlib.pyplot as plt
import numpy as np
# importing project configuration and preprocessing utilities
from . import config
from .preprocessing import load_image
# importing raw and color-constancy pipelines
from .pipelines import run_raw_pipeline, run_cc_pipeline
# importing superpixel visualization helper
from .superpixels import overlay_superpixels

# setting the object names to visualize
OBJECT_NAMES = ["apple", "deer", "cup1", "frog1"]
# selecting which lighting conditions to display
INDICES_TO_SHOW = [0, 4, 9]  # showing first, middle and last lighting variations


# creating an illumination grid figure for each object
def run(object_names=OBJECT_NAMES, indices_to_show=INDICES_TO_SHOW):
    # creating the figures directory if missing
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    # creating an illumination grid for each object
    for object_name in object_names:
        # collecting all available images for this object
        all_files = sorted(
            f for f in os.listdir(config.RAW_DIR)
            if f.startswith(object_name + "_") and f.endswith(".png"))
        # checking for missing images
        if len(all_files) == 0:
            print(f"no images for '{object_name}'")
            continue
        # selecting valid indices based on available images
        idxs = [i for i in indices_to_show if i < len(all_files)]
        chosen_files = [all_files[i] for i in idxs]
        print(f"\nCreating grid for {object_name}: {chosen_files}")
        # creating a figure with one row per selected lighting condition
        n_rows = len(chosen_files)
        fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
        # ensuring consistent 2D indexing even when only one row exists
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        # generating visualizations for each chosen image
        for row, fname in enumerate(chosen_files):
            # loading the image
            img = load_image(fname)
            # running the raw pipeline
            labels_raw = run_raw_pipeline(img)
            # running the color-constancy pipeline
            img_cc, labels_cc = run_cc_pipeline(img, cc_method="gray_world")
            # creating overlay visualizations for superpixel boundaries
            overlay_raw = overlay_superpixels(img, labels_raw)
            overlay_cc = overlay_superpixels(img_cc, labels_cc)
            # displaying the original image
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Original ({fname})")
            axes[row, 0].axis("off")
            # displaying raw superpixels
            axes[row, 1].imshow(overlay_raw)
            axes[row, 1].set_title("Raw superpixels")
            axes[row, 1].axis("off")
            # displaying CC superpixels
            axes[row, 2].imshow(overlay_cc)
            axes[row, 2].set_title("CC superpixels")
            axes[row, 2].axis("off")
        plt.tight_layout()
        # saving the figure for this object
        out_path = os.path.join(
            config.FIGURES_DIR, f"{object_name}_illumination_grid.png")
        plt.savefig(out_path, dpi=200)
        plt.show()
        print("Saved grid to:", out_path)
