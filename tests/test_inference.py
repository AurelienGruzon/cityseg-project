from __future__ import annotations

import numpy as np

from src.inference.cityseg_infer import preprocess_image, make_overlay


def test_preprocess_image_returns_expected_shapes():
    img = np.zeros((300, 600, 3), dtype=np.uint8)

    x, img_resized = preprocess_image(img, (256, 512))

    assert x.shape == (1, 256, 512, 3)
    assert x.dtype == np.float32
    assert img_resized.shape == (256, 512, 3)
    assert img_resized.dtype == np.uint8


def test_make_overlay_returns_expected_shape():
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    mask = np.zeros((256, 512), dtype=np.int32)

    overlay = make_overlay(img, mask, alpha=0.5, void_transparent=True)

    assert overlay.shape == (256, 512, 3)
    assert overlay.dtype == np.uint8