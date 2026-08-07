# SPDX-License-Identifier: Apache-2.0

"""Build the recurrent Sparse4D architecture used by the paper checkpoints.

The release checkpoints already contain complete detector state dictionaries.
Consequently, inference only needs NVIDIA's architecture configuration and
k-means anchors; it does not need to download the separate 804 MB NGC base
checkpoint before loading a SparseQID artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .sparse4d import Sparse4D

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs" / "sparse4d_rn101_v2.2"


def load_config(config_dir: str | Path):
    """Load the pinned NGC architecture config and repository anchor array."""
    config_dir = Path(config_dir)
    experiment = config_dir / "experiment.yaml"
    anchors = sorted(config_dir.glob("*.npy"))
    if not experiment.is_file() or len(anchors) != 1:
        raise FileNotFoundError(f"expected experiment.yaml and one anchor .npy under {config_dir}")
    cfg = OmegaConf.load(str(experiment))
    cfg.model.head.instance_bank.anchor = str(anchors[0])
    return cfg


def build_architecture(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    device: str | torch.device = "cuda",
):
    """Instantiate the recurrent detector without loading a base checkpoint."""
    cfg = load_config(config_dir)
    model = Sparse4D(config=cfg)
    model.to(device)
    return model, cfg


def _load_component(module, state: dict[str, Any], name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{name} state mismatch: missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)})"
        )


def load_detector_state(model: Sparse4D, checkpoint: dict[str, Any]) -> None:
    """Strictly load all detector components stored in a paper checkpoint."""
    required = ("backbone_state", "neck_state", "head_state")
    absent = [key for key in required if key not in checkpoint]
    if absent:
        raise KeyError(f"checkpoint is missing detector components: {absent}")
    _load_component(model.img_backbone, checkpoint["backbone_state"], "backbone")
    _load_component(model.img_neck, checkpoint["neck_state"], "neck")
    _load_component(model.head, checkpoint["head_state"], "head")


def build_paper_detector(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
):
    """Build and load the complete recurrent detector from a paper artifact."""
    model, cfg = build_architecture(config_dir=config_dir, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected a checkpoint dictionary, got {type(checkpoint).__name__}")
    load_detector_state(model, checkpoint)
    model.eval()
    return model, cfg, checkpoint
