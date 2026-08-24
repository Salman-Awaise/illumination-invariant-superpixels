# testing the two pipelines end to end
import numpy as np
import pytest

from src.metrics import boundary_iou, compute_stability, variation_of_information
from src.pipelines import run_cc_pipeline, run_raw_pipeline
from src.preprocessing import load_image
from src.superpixels import overlay_superpixels, run_slic


# loading one dataset image once for the integration tests
@pytest.fixture(scope="module")
def image():
    return load_image("apple_01.png")


# ------------------------------------------------------------------- determinism
# every reproducibility claim in the README rests on SLIC being deterministic, so
# that property is checked rather than assumed

def test_raw_pipeline_is_deterministic(image):
    a = run_raw_pipeline(image)
    b = run_raw_pipeline(image)
    np.testing.assert_array_equal(a, b)


def test_cc_pipeline_is_deterministic(image):
    img_a, labels_a = run_cc_pipeline(image)
    img_b, labels_b = run_cc_pipeline(image)
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_array_equal(img_a, img_b)


def test_repeated_runs_score_as_identical(image):
    # a metric comparing two runs of the same pipeline must report perfect agreement,
    # which ties the determinism property to the numbers actually reported
    a = run_raw_pipeline(image)
    b = run_raw_pipeline(image)
    assert boundary_iou(a, b) == pytest.approx(1.0, abs=1e-6)
    assert compute_stability(a, b) == pytest.approx(1.0, abs=1e-6)
    assert variation_of_information(a, b) == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------- label contract

def test_labels_are_zero_based_and_contiguous(image):
    labels = run_raw_pipeline(image)
    ids = np.unique(labels)
    assert ids.min() == 0
    # start_label=0 with no gaps is what the metrics assume when they use bincount
    np.testing.assert_array_equal(ids, np.arange(len(ids)))


def test_label_map_matches_image_geometry(image):
    labels = run_raw_pipeline(image)
    assert labels.shape == image.shape[:2]


def test_segment_count_tracks_the_request(image):
    few = run_slic(image, n_segments=50)
    many = run_slic(image, n_segments=400)
    assert len(np.unique(few)) < len(np.unique(many))


# --------------------------------------------------------------------- pipelines

def test_cc_pipeline_returns_a_corrected_image_and_matching_labels(image):
    img_cc, labels = run_cc_pipeline(image)
    assert img_cc.shape == image.shape
    assert img_cc.dtype == np.uint8
    assert labels.shape == image.shape[:2]


def test_the_two_pipelines_disagree_on_a_lit_image(image):
    # if correction changed nothing the whole experiment would be vacuous, so this
    # guards against the CC branch silently becoming a no-op.
    # note the comparison is against 0.99 rather than 1.0: boundary_iou divides by a
    # union carrying a +1e-6 epsilon, so identical inputs score just under 1.0 and a
    # naive "< 1.0" assertion would pass even for a no-op pipeline
    raw = run_raw_pipeline(image)
    img_cc, cc = run_cc_pipeline(image)
    assert not np.array_equal(raw, cc)
    assert boundary_iou(raw, cc) < 0.99


def test_cc_pipeline_actually_corrects_the_image(image):
    # guarding the same no-op risk one step earlier, on the image rather than the labels
    img_cc, _ = run_cc_pipeline(image)
    assert not np.array_equal(img_cc, image)


def test_overlay_returns_a_displayable_image(image):
    labels = run_raw_pipeline(image)
    out = overlay_superpixels(image, labels)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.max() <= 255
