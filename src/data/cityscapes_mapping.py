from __future__ import annotations

import numpy as np
from .cityscapes_labels import G, GROUPS, LUT


def map_mask_to_groups(mask_label_ids: np.ndarray) -> np.ndarray:
    """
    Convertit un mask Cityscapes labelIds (H,W) en mask groupIds (H,W) sur 8 classes (0..7).
    - tout labelId hors LUT -> void
    """
    if mask_label_ids.ndim != 2:
        raise ValueError(f"mask_label_ids must be 2D (H,W). Got shape={mask_label_ids.shape}")

    mask = mask_label_ids.astype(np.int32, copy=False)
    out = np.full(mask.shape, G["void"], dtype=np.uint8)

    valid = (mask >= 0) & (mask < LUT.shape[0])
    out[valid] = LUT[mask[valid]]
    return out


def void_ratio(mask_group_ids: np.ndarray) -> float:
    if mask_group_ids.ndim != 2:
        raise ValueError(f"mask_group_ids must be 2D (H,W). Got shape={mask_group_ids.shape}")
    return float((mask_group_ids == G["void"]).mean())


def group_hist(mask_group_ids: np.ndarray) -> np.ndarray:
    """
    Histogramme des 8 groupes. Retourne un array shape (8,) = counts.
    """
    if mask_group_ids.ndim != 2:
        raise ValueError(f"mask_group_ids must be 2D (H,W). Got shape={mask_group_ids.shape}")
    return np.bincount(mask_group_ids.ravel(), minlength=len(GROUPS))