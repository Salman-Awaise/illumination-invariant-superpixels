# keeping the small worked-example figures used in the README here
import os
import matplotlib.pyplot as plt
import numpy as np
# importing project configuration and preprocessing utilities
from . import config
from .preprocessing import load_image, apply_color_constancy
# importing raw and color-constancy pipelines
from .pipelines import run_raw_pipeline, run_cc_pipeline
# importing superpixel helpers
from .superpixels import run_slic, overlay_superpixels
# importing the reflectance ground-truth loader
from .reflectance_baseline import load_reflectance
# importing metrics used to annotate the figures
from .metrics import boundary_iou, labels_to_boundaries, variation_of_information

# setting the object and the illumination pair used for the worked example
EXAMPLE_OBJECT = "apple"
EXAMPLE_REFERENCE = "apple_01.png"
EXAMPLE_VARIANT = "apple_08.png"


# building a three-panel before and after figure for a single image
def plot_before_after(fname=EXAMPLE_VARIANT):
    # loading the image
    img = load_image(fname)
    # applying color constancy so the corrected input can be shown
    img_cc = apply_color_constancy(img, method="gray_world")
    # running both pipelines
    labels_raw = run_raw_pipeline(img)
    _, labels_cc = run_cc_pipeline(img, cc_method="gray_world")
    # drawing boundaries over each image
    overlay_raw = overlay_superpixels(img, labels_raw)
    overlay_cc = overlay_superpixels(img_cc, labels_cc)
    # laying out the three panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        (img, f"Input\n{fname}"),
        (overlay_raw, "Before: SLIC on raw RGB"),
        (overlay_cc, "After: gray-world CC, then SLIC"),]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    _save(plt, "example_before_after.png")


# building an RGB overlay that shows where two boundary maps agree and differ
def _agreement_overlay(img, labels_a, labels_b):
    # converting boundary maps from the two label images
    edges_a = labels_to_boundaries(labels_a)
    edges_b = labels_to_boundaries(labels_b)
    # dimming a grayscale copy of the image to use as the backdrop
    gray = img.astype(np.float32).mean(axis=2) / 255.0
    canvas = np.dstack([gray, gray, gray]) * 0.35
    # marking boundaries present in only one of the two segmentations
    only_a = np.logical_and(edges_a, np.logical_not(edges_b))
    only_b = np.logical_and(edges_b, np.logical_not(edges_a))
    shared = np.logical_and(edges_a, edges_b)
    # painting reference-only boundaries red and variant-only boundaries cyan
    canvas[only_a] = [1.0, 0.25, 0.25]
    canvas[only_b] = [0.25, 0.85, 1.0]
    # painting shared boundaries white
    canvas[shared] = [1.0, 1.0, 1.0]
    return canvas


# building a figure that visualises the boundary agreement behind the IoU numbers
def plot_boundary_agreement(ref_name=EXAMPLE_REFERENCE, var_name=EXAMPLE_VARIANT):
    # loading the reference and the variant under different lighting
    img_ref = load_image(ref_name)
    img_var = load_image(var_name)
    # segmenting both with the raw pipeline
    ref_raw = run_raw_pipeline(img_ref)
    var_raw = run_raw_pipeline(img_var)
    # segmenting both with the color-constancy pipeline
    _, ref_cc = run_cc_pipeline(img_ref, cc_method="gray_world")
    _, var_cc = run_cc_pipeline(img_var, cc_method="gray_world")
    # scoring each pipeline on this particular pair
    iou_raw = boundary_iou(ref_raw, var_raw)
    iou_cc = boundary_iou(ref_cc, var_cc)
    # building the two overlays
    overlay_raw = _agreement_overlay(img_ref, ref_raw, var_raw)
    overlay_cc = _agreement_overlay(img_ref, ref_cc, var_cc)
    # laying out the two panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(overlay_raw)
    axes[0].set_title(f"Before: raw RGB\nboundary IoU = {iou_raw:.3f}")
    axes[1].imshow(overlay_cc)
    axes[1].set_title(f"After: gray-world CC\nboundary IoU = {iou_cc:.3f}")
    for ax in axes:
        ax.axis("off")
    # explaining the color coding under the panels
    fig.suptitle(
        f"Boundary agreement between {ref_name} and {var_name}\n"
        "white = boundaries both segmentations agree on, "
        "red = reference only, blue = variant only",
        fontsize=10)
    plt.tight_layout()
    _save(plt, "example_boundary_agreement.png")


# building a figure comparing both pipelines against the reflectance ground truth
def plot_reflectance_example(object_name=EXAMPLE_OBJECT, fname=EXAMPLE_VARIANT):
    # loading the image and the illumination-free reflectance ground truth
    img = load_image(fname)
    refl = load_reflectance(object_name)
    # segmenting the reflectance image to get the target partition
    labels_refl = run_slic(refl)
    # running both pipelines on the lit image
    labels_raw = run_raw_pipeline(img)
    img_cc, labels_cc = run_cc_pipeline(img, cc_method="gray_world")
    # scoring both pipelines against the reflectance partition
    iou_raw = boundary_iou(labels_raw, labels_refl)
    iou_cc = boundary_iou(labels_cc, labels_refl)
    # laying out the three panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    panels = [
        (overlay_superpixels(refl, labels_refl), "Target: SLIC on GT reflectance"),
        (overlay_superpixels(img, labels_raw), f"Before: raw RGB\nIoU vs target = {iou_raw:.3f}"),
        (overlay_superpixels(img_cc, labels_cc), f"After: gray-world CC\nIoU vs target = {iou_cc:.3f}"),]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    _save(plt, "example_reflectance.png")


# generating every worked-example figure
def run():
    # creating the figures directory if missing
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_before_after()
    plot_boundary_agreement()
    plot_reflectance_example()


# saving the current figure into FIGURES_DIR
def _save(plt_module, filename):
    out_path = os.path.join(config.FIGURES_DIR, filename)
    plt_module.savefig(out_path, dpi=160, bbox_inches="tight")
    plt_module.show()
    print("Saved figure to:", out_path)
