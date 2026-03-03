from __future__ import annotations
from cityscapesscripts.helpers.labels import labels

import numpy as np

GROUPS = ["flat", "human", "vehicle", "construction", "object", "nature", "sky", "void"]
G = {name: i for i, name in enumerate(GROUPS)}

def build_id2group_and_lut() -> tuple[dict[int, str], np.ndarray]:
    # Source canonique Cityscapes (pip: cityscapesScripts)
    from cityscapesscripts.helpers.labels import labels  # type: ignore

    id2group: dict[int, str] = {}
    max_id = -1

    for lbl in labels:
        # lbl.id correspond aux labelIds dans gtFine/*labelIds.png
        if lbl.id == -1:
            continue
        max_id = max(max_id, lbl.id)
        # lbl.category est l'un des 8 groupes (flat/human/...)
        id2group[lbl.id] = lbl.category

    if max_id < 0:
        raise RuntimeError("Impossible de construire le LUT: max_id invalide")

    lut = np.full((max_id + 1,), G["void"], dtype=np.uint8)
    for k, cat in id2group.items():
        # sécurité : si jamais une catégorie inconnue apparaît
        if cat not in G:
            raise ValueError(f"Catégorie inattendue dans labels.py: {cat}")
        lut[k] = G[cat]

    return id2group, lut

ID2GROUP, LUT = build_id2group_and_lut()
MAX_LABEL_ID = int(LUT.shape[0] - 1)