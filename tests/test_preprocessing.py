# testing image loading and the gray-world correction
import numpy as np
import pytest

from src.preprocessing import apply_color_constancy, gray_world_cc, load_image


# building an image whose channel means are already equal
@pytest.fixture
def neutral_image():
    return np.full((8, 8, 3), 120, dtype=np.uint8)


# building an image with a strong red cast
@pytest.fixture
def red_cast_image():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = 90
    img[..., 2] = 60
    return img


# --------------------------------------------------------------- output contract
# downstream code hands the result straight to SLIC, so dtype and range matter

def test_output_is_uint8_in_range(red_cast_image):
    out = gray_world_cc(red_cast_image)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert out.shape == red_cast_image.shape


def test_input_is_not_mutated(red_cast_image):
    # the pipelines reuse the loaded image for the raw condition after correcting it
    # for the CC condition, so in-place modification would silently couple the two
    before = red_cast_image.copy()
    gray_world_cc(red_cast_image)
    np.testing.assert_array_equal(red_cast_image, before)


# ------------------------------------------------------------------- correctness

def test_already_neutral_image_is_left_effectively_unchanged(neutral_image):
    # equal channel means give a scale factor of one, so only rounding may move a
    # value, and then by at most a single level
    out = gray_world_cc(neutral_image)
    assert np.abs(out.astype(np.int16) - neutral_image.astype(np.int16)).max() <= 1


def test_colour_cast_is_reduced(red_cast_image):
    # the point of the correction is to bring the channel means together
    before = red_cast_image.reshape(-1, 3).mean(axis=0)
    after = gray_world_cc(red_cast_image).reshape(-1, 3).mean(axis=0)
    assert after.std() < before.std()
    # and the corrected means should be close to equal
    assert after.std() < 2.0


def test_correction_preserves_flat_regions(red_cast_image):
    # a constant-colour input must stay constant, since every pixel is scaled alike
    out = gray_world_cc(red_cast_image)
    for c in range(3):
        assert out[..., c].min() == out[..., c].max()


# ------------------------------------------------------------------- error paths

def test_unknown_method_raises():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        apply_color_constancy(img, method="white_patch")


def test_gray_world_is_the_dispatched_default(red_cast_image):
    np.testing.assert_array_equal(
        apply_color_constancy(red_cast_image), gray_world_cc(red_cast_image))


def test_missing_image_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("this_image_does_not_exist.png")


# ------------------------------------------------------------------ real dataset

def test_loading_a_dataset_image_gives_rgb_uint8():
    img = load_image("apple_01.png")
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    assert np.isfinite(img).all()
