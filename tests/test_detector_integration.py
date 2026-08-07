from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sparseqid.constants import CLASS_ID_TO_NAME
from sparseqid.detector import DEFAULT_CONFIG_DIR, build_paper_detector
from sparseqid.detector.build import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_CHECKPOINT = REPO_ROOT / "checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth"


def test_pinned_detector_config_and_anchor_contract():
    cfg = load_config(DEFAULT_CONFIG_DIR)
    anchors = np.load(next(DEFAULT_CONFIG_DIR.glob("*.npy")))

    assert cfg.model.head.instance_bank.num_anchor == 900
    assert anchors.shape == (900, 11)
    assert cfg.model.head.instance_bank.num_temp_instances == 600
    assert "temp_gnn" in cfg.model.head.operation_order
    assert cfg.model.head.deformable_model.num_levels == 4


def test_aicity_2026_class_map():
    assert CLASS_ID_TO_NAME == {
        0: "Person",
        1: "Forklift",
        2: "NovaCarter",
        3: "Transporter",
        4: "FourierGR1T2",
        5: "AgilityDigit",
        6: "PalletTruck",
    }


@pytest.mark.weights
@pytest.mark.skipif(
    os.environ.get("SPARSEQID_RUN_WEIGHT_TESTS") != "1",
    reason="set SPARSEQID_RUN_WEIGHT_TESTS=1 to load the full paper detector",
)
def test_submission_checkpoint_strictly_loads_without_ngc_base():
    assert SUBMISSION_CHECKPOINT.is_file()
    model, _cfg, checkpoint = build_paper_detector(SUBMISSION_CHECKPOINT, device="cpu")

    assert checkpoint["iter"] == 6000
    assert checkpoint["vocab"] == 128
    assert model.head.instance_bank.anchor.shape == (900, 11)
