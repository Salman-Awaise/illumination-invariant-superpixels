# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping superpixel processing pipelines here
from .preprocessing import apply_color_constancy
from .superpixels import run_slic
# running superpixels on the raw RGB image
def run_raw_pipeline(img, n_segments=200, compactness=10.0):
    # running superpixels on the raw RGB image
    labels = run_slic(img, n_segments=n_segments, compactness=compactness)
    return labels
# running color constancy followed by superpixels
def run_cc_pipeline(img, n_segments=200, compactness=10.0, cc_method="gray_world"):
    # applying color constancy correction to the image
    img_cc = apply_color_constancy(img, method=cc_method)
    # running superpixels on the color corrected image
    labels = run_slic(img_cc, n_segments=n_segments, compactness=compactness)
    return img_cc, labels
