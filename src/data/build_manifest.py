#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Pair:
    split: str
    city: str
    image_path: Path
    mask_path: Path


def iter_pairs(data_root: Path, split: str) -> Iterable[Pair]:
    """
    Expects:
      data_root/leftImg8bit/<split>/<city>/*_leftImg8bit.png
      data_root/gtFine/<split>/<city>/*_gtFine_labelIds.png

    Pairs are matched by the shared stem:
      <city>_<seq>_<frame>_
    """
    images_base = data_root / "leftImg8bit" / split
    masks_base = data_root / "gtFine" / split

    if not images_base.exists():
        raise FileNotFoundError(f"Missing images folder: {images_base}")
    if not masks_base.exists():
        raise FileNotFoundError(f"Missing masks folder: {masks_base}")

    for city_dir in sorted(p for p in images_base.iterdir() if p.is_dir()):
        city = city_dir.name

        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            stem = img_path.name.replace("_leftImg8bit.png", "")  # <city>_<seq>_<frame>
            mask_path = masks_base / city / f"{stem}_gtFine_labelIds.png"

            # Many setups have no public labels for "test" split.
            if not mask_path.exists():
                continue

            yield Pair(split=split, city=city, image_path=img_path, mask_path=mask_path)


def write_manifest(pairs: List[Pair], out_csv: Path, data_root: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        return p.relative_to(data_root).as_posix()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "city", "image_path", "mask_path"])
        for p in pairs:
            w.writerow([p.split, p.city, rel(p.image_path), rel(p.mask_path)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Cityscapes manifest CSV (image/mask pairs).")
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Folder containing leftImg8bit/ and gtFine/")
    ap.add_argument("--out", type=Path, default=Path("data/manifests/cityscapes_pairs.csv"), help="Output CSV path")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to index (train val test)")
    args = ap.parse_args()

    data_root = args.data_root.resolve()

    pairs: List[Pair] = []
    for sp in args.splits:
        pairs.extend(list(iter_pairs(data_root, sp)))

    if not pairs:
        raise RuntimeError(
            "No (image,mask) pairs found.\n"
            f"Checked data_root={data_root}\n"
            "Expected:\n"
            "  leftImg8bit/<split>/<city>/*_leftImg8bit.png\n"
            "  gtFine/<split>/<city>/*_gtFine_labelIds.png\n"
        )

    write_manifest(pairs, args.out, data_root)

    # Report
    counts = {}
    for p in pairs:
        counts[p.split] = counts.get(p.split, 0) + 1

    print(f"✅ Manifest written: {args.out} ({len(pairs)} pairs)")
    for sp in sorted(counts):
        print(f"  - {sp}: {counts[sp]}")
    print("Sample:")
    print("  image:", pairs[0].image_path)
    print("  mask :", pairs[0].mask_path)


if __name__ == "__main__":
    main()