# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping helper functions for saving images and label maps
import os
import numpy as np
import cv2
from .config import RESULTS_DIR
# saving RGB image to disk 
def save_rgb_image(img, out_path):
    # creating output directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # converting RGB to BGR for OpenCV 
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # writing image to disk
    cv2.imwrite(out_path, img_bgr)
# saving superpixel label map as .npy
def save_label_map(labels, out_path):
    # creating output directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # saving label map as numpy file
    np.save(out_path, labels)
# loading a superpixel label map
def load_label_map(path):
    labels = np.load(path)
    return labels
