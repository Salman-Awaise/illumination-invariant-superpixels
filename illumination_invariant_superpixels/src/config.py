# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping common paths and default settings here
import os
# getting the directory of the file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# getting the project root 
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
# setting up data folder paths 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
GT_DIR = os.path.join(DATA_DIR, "gt")
# setting up results folder paths 
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
# setting default superpixel parameters for SLIC segmentation
DEFAULT_N_SEGMENTS = 200
DEFAULT_COMPACTNESS = 10.0
