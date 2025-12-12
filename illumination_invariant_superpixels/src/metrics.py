# Author: Salman Awaise, Sameer Syed
# Date: December 11, 2025
# Course: CS 7180 - Advanced Perception
# keeping metric functions for superpixel-based evaluation here
import numpy as np
# computing boundary recall between predicted and ground-truth edges
def compute_boundary_recall(pred_edges, gt_edges):
    # converting to boolean arrays for logical operations
    pred_edges = pred_edges.astype(bool)
    gt_edges = gt_edges.astype(bool)
    # computing true positives as intersection of predicted and ground-truth edges
    tp = np.logical_and(pred_edges, gt_edges).sum()
    # computing total ground-truth edge count 
    total_gt = gt_edges.sum() + 1e-6
    # computing boundary recall as ratio of true positives to total ground-truth edges
    br = tp / total_gt
    return float(br)
# computing Achievable Segmentation Accuracy between superpixel labels and ground-truth regions
def compute_ASA(labels, gt_regions):
    # flattening label maps to 1D arrays for easier processing
    labels_flat = labels.reshape(-1).astype(np.int64)
    gt_flat = gt_regions.reshape(-1).astype(np.int64)
    # initializing ASA accumulator
    asa_sum = 0.0
    total_pixels = len(labels_flat)
    # iterating over each unique superpixel ID
    for sp_id in np.unique(labels_flat):
        # creating mask for pixels belonging to this superpixel
        mask = labels_flat == sp_id
        # getting ground-truth region IDs within this superpixel
        gt_in_sp = gt_flat[mask]
        # skipping if superpixel is empty
        if gt_in_sp.size == 0:
            continue
        # finding the most common ground-truth region within this superpixel
        max_count = np.bincount(gt_in_sp).max()
        # adding to accumulator
        asa_sum += max_count
    # computing final ASA as ratio of correctly assigned pixels to total pixels
    asa = asa_sum / (total_pixels + 1e-6)
    return float(asa)
# computing neighbor based stability score between two label maps
def compute_stability(labels_1, labels_2):
    # checking if label maps have the same shape
    if labels_1.shape != labels_2.shape:
        raise ValueError("label maps must have the same shape")
    # getting image dimensions
    h, w = labels_1.shape
    # extracting horizontal neighbor pairs from both label maps
    l1_h = labels_1[:, :-1]
    l2_h = labels_2[:, :-1]
    l1_h_next = labels_1[:, 1:]
    l2_h_next = labels_2[:, 1:]
    # checking which horizontal neighbors are in the same superpixel
    same_l1_h = l1_h == l1_h_next
    same_l2_h = l2_h == l2_h_next
    # extracting vertical neighbor pairs from both label maps
    l1_v = labels_1[:-1, :]
    l2_v = labels_2[:-1, :]
    l1_v_next = labels_1[1:, :]
    l2_v_next = labels_2[1:, :]
    # checking which vertical neighbors are in the same superpixel
    same_l1_v = l1_v == l1_v_next
    same_l2_v = l2_v == l2_v_next
    # computing total number of same-superpixel pairs in labels_1
    total_same_l1 = same_l1_h.sum() + same_l1_v.sum() + 1e-6
    # counting how many pairs stay together in both segmentations
    agree_h = np.logical_and(same_l1_h, same_l2_h).sum()
    agree_v = np.logical_and(same_l1_v, same_l2_v).sum()
    agree_total = agree_h + agree_v
    # computing stability as ratio of agreements to total pairs
    stability = agree_total / total_same_l1
    return float(stability)
# converting superpixel label map into binary boundary map
def labels_to_boundaries(labels):
    # getting image dimensions
    h, w = labels.shape
    # initializing boundary map as all zeros
    boundaries = np.zeros((h, w), dtype=bool)
    # marking horizontal boundaries where adjacent pixels have different labels
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    # marking vertical boundaries where adjacent pixels have different labels
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundaries
# computing Intersection over Union between boundaries of two superpixel segmentations
def boundary_iou(labels_1, labels_2):
    # checking if label maps have the same shape
    if labels_1.shape != labels_2.shape:
        raise ValueError("label maps must have same shape for boundary IoU")
    # converting label maps to boundary maps
    edges_1 = labels_to_boundaries(labels_1)
    edges_2 = labels_to_boundaries(labels_2)
    # computing intersection of boundaries
    inter = np.logical_and(edges_1, edges_2).sum()
    # computing union of boundaries
    union = np.logical_or(edges_1, edges_2).sum() + 1e-6
    # computing IoU as ratio of intersection to union
    iou = inter / union
    return float(iou)
# computing entropy from 1D count array
def _entropy_from_counts(counts):
    # converting to float for probability calculations
    counts = counts.astype(np.float64)
    # computing total count
    total = counts.sum()
    # returning zero entropy if no counts
    if total <= 0:
        return 0.0
    # computing probabilities
    p = counts / total
    # filtering out zero probabilities
    p = p[p > 0]
    # computing entropy using Shannon formula
    return float(-np.sum(p * np.log2(p)))
# computing Variation of Information between two segmentations
def variation_of_information(labels_1, labels_2):
    # checking if label maps have the same shape
    if labels_1.shape != labels_2.shape:
        raise ValueError("label maps must have same shape for VI")
    # flattening label maps to 1D arrays
    x = labels_1.reshape(-1).astype(np.int64)
    y = labels_2.reshape(-1).astype(np.int64)
    # reindexing labels to consecutive IDs
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    # computing maximum label values for array sizing
    max_x = int(x.max()) + 1
    max_y = int(y.max()) + 1
    # computing marginal counts for each segmentation
    count_x = np.bincount(x, minlength=max_x)
    count_y = np.bincount(y, minlength=max_y)
    # computing joint counts by creating unique index for each pair
    joint_index = x * max_y + y
    count_xy = np.bincount(joint_index, minlength=max_x * max_y)
    count_xy = count_xy.reshape(max_x, max_y)
    # computing marginal entropies
    Hx = _entropy_from_counts(count_x)
    Hy = _entropy_from_counts(count_y)
    # computing total number of pixels
    total = float(count_xy.sum())
    # returning zero if no pixels
    if total <= 0:
        return 0.0
    # computing joint and marginal probability distributions
    p_xy = count_xy / total
    p_x = count_x / total
    p_y = count_y / total
    # creating mask for nonzero joint probabilities
    mask = p_xy > 0
    p_xy_nonzero = p_xy[mask]
    x_idx, y_idx = np.nonzero(mask)
    # computing mutual information 
    mi = np.sum(
        p_xy_nonzero
        * (np.log2(p_xy_nonzero) - np.log2(p_x[x_idx]) - np.log2(p_y[y_idx]))
    )
    # computing Variation of Information
    vi = Hx + Hy - 2.0 * mi
    return float(vi)
