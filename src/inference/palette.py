# src/inference/palette.py
from __future__ import annotations

import numpy as np

# Index = class id (0..7) dans l'ordre GROUPS = ["flat","human","vehicle","construction","object","nature","sky","void"]
# Couleurs simples (RGB). Tu peux les changer plus tard.
PALETTE_RGB = np.array([
    [128,  64, 128],  # flat
    [220,  20,  60],  # human
    [  0,   0, 142],  # vehicle
    [ 70,  70,  70],  # construction
    [250, 170,  30],  # object
    [107, 142,  35],  # nature
    [ 70, 130, 180],  # sky
    [  0,   0,   0],  # void (souvent transparent en overlay)
], dtype=np.uint8)

VOID_ID = 7

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) int in [0..7]
    returns: (H,W,3) uint8
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")
    return PALETTE_RGB[mask]