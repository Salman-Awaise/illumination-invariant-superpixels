# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping image loading and color constancy functions here
import os
import cv2
import numpy as np
from . import config
# loading RGB image from RAW_DIR
def load_image(image_name):
    # building full path using RAW_DIR from config
    image_path = os.path.join(config.RAW_DIR, image_name)
    # reading BGR image using OpenCV
    img_bgr = cv2.imread(image_path)
    # checking if image exists
    if img_bgr is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    # converting BGR to RGB for consistency
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
# applying gray-world color constancy
def gray_world_cc(img):
    # converting to float for scaling
    img = img.astype(np.float32)
    # computing channel means
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    # computing gray target value
    gray_value = (mean_r + mean_g + mean_b) / 3.0
    # setting small epsilon to avoid division by zero
    eps = 1e-6
    # scaling channels toward the gray value
    img[:, :, 0] *= gray_value / (mean_r + eps)
    img[:, :, 1] *= gray_value / (mean_g + eps)
    img[:, :, 2] *= gray_value / (mean_b + eps)
    # clipping values and converting to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
# applying color constancy method
def apply_color_constancy(img, method="gray_world"):
    if method == "gray_world":
        return gray_world_cc(img)
    else:
        # handling unsupported methods
        raise NotImplementedError(f"color constancy method '{method}' not implemented yet")
