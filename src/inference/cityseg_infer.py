# src/inference/cityseg_infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf

from .palette import colorize_mask, VOID_ID


@dataclass(frozen=True)
class InferConfig:
    target_hw: Tuple[int, int] = (256, 512)  # (H,W) doit matcher le training
    alpha: float = 0.5                       # intensité overlay
    void_transparent: bool = True            # void pas affiché sur overlay


def load_keras_model(model_path: str | Path) -> tf.keras.Model:
    model_path = str(model_path)
    return tf.keras.models.load_model(model_path, compile=False)


def load_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess_image(img_rgb: np.ndarray, target_hw: Tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    img_rgb: (H,W,3) uint8
    returns:
      x: (1, th, tw, 3) float32 in [0,1]
      orig: original image uint8 for later overlay
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be (H,W,3), got {img_rgb.shape}")

    th, tw = target_hw
    pil = Image.fromarray(img_rgb)
    pil_resized = pil.resize((tw, th), resample=Image.BILINEAR)
    x = np.array(pil_resized, dtype=np.float32) / 255.0
    x = x[None, ...]  # batch dim
    return x, np.array(pil_resized, dtype=np.uint8)


def predict_mask(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    """
    x: (1,H,W,3) float32
    returns mask: (H,W) int32 in [0..7]
    """
    logits = model.predict(x, verbose=0)  # (1,H,W,C)
    if logits.ndim != 4:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    mask = np.argmax(logits[0], axis=-1).astype(np.int32)
    return mask


def make_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.5,
    void_transparent: bool = True,
) -> np.ndarray:
    """
    img_rgb: (H,W,3) uint8
    mask: (H,W) int
    returns overlay: (H,W,3) uint8
    """
    if img_rgb.shape[:2] != mask.shape:
        raise ValueError(f"img and mask must match spatially: img={img_rgb.shape}, mask={mask.shape}")

    colored = colorize_mask(mask)  # (H,W,3)
    img_f = img_rgb.astype(np.float32)
    col_f = colored.astype(np.float32)

    if void_transparent:
        m = (mask != VOID_ID)[..., None].astype(np.float32)  # 1 where not void
    else:
        m = np.ones((*mask.shape, 1), dtype=np.float32)

    out = img_f * (1.0 - alpha * m) + col_f * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def run_inference(
    model_path: str | Path,
    image_path: str | Path,
    out_dir: str | Path,
    cfg: Optional[InferConfig] = None,
) -> dict:
    cfg = cfg or InferConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_keras_model(model_path)

    img0 = load_image_rgb(image_path)
    x, img_resized = preprocess_image(img0, cfg.target_hw)

    mask = predict_mask(model, x)
    mask_rgb = colorize_mask(mask)
    overlay = make_overlay(img_resized, mask, alpha=cfg.alpha, void_transparent=cfg.void_transparent)

    stem = Path(image_path).stem.replace("_leftImg8bit", "")
    p_mask = out_dir / f"{stem}_pred_mask.png"
    p_over = out_dir / f"{stem}_pred_overlay.png"

    Image.fromarray(mask_rgb).save(p_mask)
    Image.fromarray(overlay).save(p_over)

    return {
        "mask_path": str(p_mask),
        "overlay_path": str(p_over),
        "target_hw": cfg.target_hw,
    }