# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping SLIC superpixel logic and visualization here
import numpy as np
from skimage.segmentation import slic, mark_boundaries
# running SLIC superpixel segmentation
def run_slic(img, n_segments=200, compactness=10.0):
    # running SLIC on the RGB image to generate superpixel labels
    labels = slic(img,n_segments=n_segments,compactness=compactness,start_label=0)
    return labels
# overlaying superpixel boundaries on the image
def overlay_superpixels(img, labels):
    # drawing superpixel boundaries over the image
    boundary_img = mark_boundaries(img, labels)
    # converting normalized float image to uint8 format
    boundary_img = (boundary_img * 255).astype(np.uint8)
    return boundary_img
