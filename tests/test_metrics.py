# testing the evaluation metrics against properties that must hold by definition
import numpy as np
import pytest

from src.metrics import (
    boundary_iou,
    compute_ASA,
    compute_boundary_recall,
    compute_stability,
    labels_to_boundaries,
    variation_of_information,
)


# building a small deterministic label map for the known-answer tests
@pytest.fixture
def labels_2x3_blocks():
    # two horizontal bands, three vertical columns -> six labelled blocks
    return np.array([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [3, 3, 4, 4, 5, 5],
        [3, 3, 4, 4, 5, 5],
    ], dtype=np.int64)


# ---------------------------------------------------------------- identity cases
# a segmentation compared against itself is the strongest available sanity check:
# it must score perfectly, and a sign or indexing error would break it immediately

def test_boundary_iou_of_identical_labels_is_one(labels_2x3_blocks):
    # the epsilon in the union keeps this just under 1.0 rather than exactly 1.0
    assert boundary_iou(labels_2x3_blocks, labels_2x3_blocks) == pytest.approx(1.0, abs=1e-6)


def test_stability_of_identical_labels_is_one(labels_2x3_blocks):
    assert compute_stability(labels_2x3_blocks, labels_2x3_blocks) == pytest.approx(1.0, abs=1e-6)


def test_variation_of_information_of_identical_labels_is_zero(labels_2x3_blocks):
    # identical partitions share all their information, so VI collapses to zero
    assert variation_of_information(labels_2x3_blocks, labels_2x3_blocks) == pytest.approx(0.0, abs=1e-9)


def test_variation_of_information_is_symmetric():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 8, size=(20, 25))
    b = rng.integers(0, 8, size=(20, 25))
    assert variation_of_information(a, b) == pytest.approx(variation_of_information(b, a))


def test_boundary_iou_is_symmetric():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 8, size=(20, 25))
    b = rng.integers(0, 8, size=(20, 25))
    assert boundary_iou(a, b) == pytest.approx(boundary_iou(b, a))


# ------------------------------------------------------------------------- bounds
# every metric has a defined range, and drifting outside it would invalidate any
# comparison built on top of it

def test_bounded_metrics_stay_in_range():
    rng = np.random.default_rng(2)
    for _ in range(10):
        a = rng.integers(0, 12, size=(16, 18))
        b = rng.integers(0, 12, size=(16, 18))
        assert 0.0 <= boundary_iou(a, b) <= 1.0
        assert 0.0 <= compute_stability(a, b) <= 1.0
        assert 0.0 <= compute_ASA(a, b) <= 1.0
        # VI is unbounded above but can never be negative
        assert variation_of_information(a, b) >= 0.0


def test_disjoint_partitions_score_worse_than_identical_ones():
    # one label everywhere versus a fine grid: the two partitions share no structure
    coarse = np.zeros((12, 12), dtype=np.int64)
    fine = np.arange(144, dtype=np.int64).reshape(12, 12)
    assert variation_of_information(coarse, fine) > variation_of_information(fine, fine)
    assert boundary_iou(coarse, fine) < boundary_iou(fine, fine)


# ------------------------------------------------------------------- error paths
# the shape guards are written but never exercised by the experiments, so they are
# checked here rather than assumed

@pytest.mark.parametrize("fn", [compute_stability, boundary_iou, variation_of_information])
def test_mismatched_shapes_raise(fn):
    a = np.zeros((4, 5), dtype=np.int64)
    b = np.zeros((5, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        fn(a, b)


# ------------------------------------------------------------------ known answers

def test_labels_to_boundaries_matches_hand_worked_result():
    # a single vertical seam between two labels: boundary pixels land on the right
    # of the seam, because the difference is recorded at the later index
    labels = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1]], dtype=np.int64)
    expected = np.array([[False, False, True, False],
                        [False, False, True, False]])
    np.testing.assert_array_equal(labels_to_boundaries(labels), expected)


def test_labels_to_boundaries_of_uniform_map_is_empty():
    labels = np.zeros((5, 5), dtype=np.int64)
    assert not labels_to_boundaries(labels).any()


def test_asa_is_one_when_superpixels_nest_inside_ground_truth():
    # every superpixel sits entirely within a single ground-truth region, which is
    # the case ASA is defined to score perfectly
    gt = np.array([[0, 0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 1, 1]], dtype=np.int64)
    labels = np.array([[0, 0, 1, 2, 2, 3],
                       [0, 0, 1, 2, 2, 3]], dtype=np.int64)
    assert compute_ASA(labels, gt) == pytest.approx(1.0, abs=1e-6)


def test_boundary_recall_recovers_all_edges_when_predictions_match():
    gt_edges = np.array([[False, True, False],
                         [False, True, False]])
    assert compute_boundary_recall(gt_edges, gt_edges) == pytest.approx(1.0, abs=1e-6)
    # and recovers none when the prediction is empty
    assert compute_boundary_recall(np.zeros_like(gt_edges), gt_edges) == pytest.approx(0.0)


def test_stability_penalises_a_split_reference_group():
    # every pixel grouped together, versus the same image split down the middle:
    # the pairs that straddle the split stop agreeing
    together = np.zeros((4, 4), dtype=np.int64)
    split = np.array([[0, 0, 1, 1]] * 4, dtype=np.int64)
    assert compute_stability(together, split) < 1.0


# --------------------------------------------------------------- degenerate cases
# these document real edge behaviour rather than asserting a desirable property, so
# that a future change to the epsilon handling is a visible decision

def test_stability_of_an_all_singleton_reference_is_zero():
    # compute_stability normalises by the pairs grouped together in the reference.
    # if the reference groups nothing, there is nothing to agree about, and the
    # epsilon in the denominator makes the result 0.0 rather than 1.0 or NaN.
    singletons = np.arange(64, dtype=np.int64).reshape(8, 8)
    assert compute_stability(singletons, singletons) == pytest.approx(0.0)


def test_identity_scores_are_epsilon_short_of_exactly_one(labels_2x3_blocks):
    # every ratio metric carries a +1e-6 guard in its denominator, so a perfect match
    # lands just below 1.0. Downstream comparisons must not test for equality with 1.
    iou = boundary_iou(labels_2x3_blocks, labels_2x3_blocks)
    assert iou < 1.0
    assert iou == pytest.approx(1.0, abs=1e-6)
