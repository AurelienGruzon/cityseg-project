# scripts/infer.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.cityseg_infer import run_inference, InferConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .keras model (best)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--h", type=int, default=256, help="Target height")
    ap.add_argument("--w", type=int, default=512, help="Target width")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha")
    ap.add_argument("--void-transparent", action="store_true", help="Do not overlay void pixels")
    args = ap.parse_args()

    cfg = InferConfig(
        target_hw=(args.h, args.w),
        alpha=args.alpha,
        void_transparent=args.void_transparent,
    )

    res = run_inference(
        model_path=args.model,
        image_path=args.image,
        out_dir=args.out,
        cfg=cfg,
    )

    print("Saved:")
    print(" - mask   :", res["mask_path"])
    print(" - overlay:", res["overlay_path"])


if __name__ == "__main__":
    main()