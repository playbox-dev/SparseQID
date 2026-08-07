"""Canonical class definitions for the AI City 2026 warehouse MTMC challenge.

The submission's ``class_id`` column MUST use ``CLASS_ID_TO_NAME`` below. This is the
**official 2026** map, verified against the dataset's ``all_txt_gt.txt`` (per-class
instance counts matched 1:1). ``tests/test_classid_map.py`` pins it.

Difference from the 2025 harness (do not regress): 2026 id ``6`` is **PalletTruck**
(2025 used ``Crate``), and 2025's static ids ``7-10`` (Basket/KLTBin/Cone/Rack) do not
exist in the 2026 data. All 7 classes are dynamic, tracked agents.
"""

from __future__ import annotations

# Official AI City 2026 class-id map (source: MTMC_Tracking_2026/all_txt_gt.txt).
CLASS_ID_TO_NAME: dict[int, str] = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}

NAME_TO_CLASS_ID: dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# All 7 classes are dynamic, tracked agents in 2026 (there are no static-object classes).
DYNAMIC_CLASS_NAMES: tuple[str, ...] = (
    "Person",
    "Forklift",
    "NovaCarter",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "PalletTruck",
)
DYNAMIC_CLASS_IDS: tuple[int, ...] = tuple(NAME_TO_CLASS_ID[n] for n in DYNAMIC_CLASS_NAMES)

NUM_CLASSES = len(DYNAMIC_CLASS_NAMES)

# The network output follows experiment.yaml rather than evaluation-id order.
MODEL_IDX_TO_CLASS_ID: dict[int, int] = {0: 0, 1: 4, 2: 5, 3: 2, 4: 3, 5: 1, 6: 6}
CLASS_ID_TO_MODEL_IDX: dict[int, int] = {v: k for k, v in MODEL_IDX_TO_CLASS_ID.items()}
NAME_TO_MODEL_IDX: dict[str, int] = {
    CLASS_ID_TO_NAME[class_id]: model_index
    for model_index, class_id in MODEL_IDX_TO_CLASS_ID.items()
}
