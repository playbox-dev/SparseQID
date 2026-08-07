# SPDX-License-Identifier: Apache-2.0

"""Construct the complete detector and identity model from release artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .detector.build import DEFAULT_CONFIG_DIR, build_architecture, load_detector_state
from .identity import IDDecoder, PosEncoder, TrajectoryModeling

FEATURE_DIM = 256


def build_identity(
    checkpoint: dict[str, Any] | None = None,
    *,
    vocabulary_size: int = 128,
    decoder_layers: int = 6,
    memory_length: int = 30,
    position_encoding: str = "fourier",
    device: str | torch.device = "cuda",
):
    """Build identity modules, optionally loading their checkpoint state."""

    if checkpoint is not None:
        vocabulary_size = int(checkpoint.get("vocab", vocabulary_size))
        decoder_state = checkpoint["id_decoder"]
        decoder_layers, memory_length = decoder_state["rel_pos_embeds"].shape[:2]
        position_encoding = checkpoint.get("pos_enc_mode", position_encoding)
    memory = TrajectoryModeling(detr_dim=FEATURE_DIM, ffn_dim_ratio=2, feature_dim=FEATURE_DIM).to(
        device
    )
    decoder = IDDecoder(
        feature_dim=FEATURE_DIM,
        id_dim=256,
        ffn_dim_ratio=2,
        num_layers=int(decoder_layers),
        head_dim=32,
        num_id_vocabulary=vocabulary_size,
        rel_pe_length=int(memory_length),
        use_aux_loss=True,
        use_shared_aux_head=True,
    ).to(device)
    position = PosEncoder(position_encoding, FEATURE_DIM).to(device)
    if checkpoint is not None:
        memory.load_state_dict(checkpoint["trajectory_modeling"])
        decoder.load_state_dict(checkpoint["id_decoder"])
        if checkpoint.get("pos_enc"):
            position.load_state_dict(checkpoint["pos_enc"])
    return memory, decoder, position, vocabulary_size


def load_base_detector_state(model, checkpoint: dict[str, Any]) -> None:
    """Load either an NGC full-model checkpoint or SparseQID component states."""

    if all(key in checkpoint for key in ("backbone_state", "neck_state", "head_state")):
        load_detector_state(model, checkpoint)
        return
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_unexpected = {key for key in unexpected if key.startswith("depth_branch.")}
    if set(unexpected) - allowed_unexpected or missing:
        raise RuntimeError(
            f"base detector state mismatch: missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)})"
        )


def load_pipeline(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
):
    """Load the complete SparseQID inference pipeline."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected a checkpoint dictionary, got {type(checkpoint).__name__}")
    detector, config = build_architecture(config_dir=config_dir, device=device)
    load_detector_state(detector, checkpoint)
    memory, decoder, position, vocabulary_size = build_identity(checkpoint, device=device)
    detector.eval()
    memory.eval()
    decoder.eval()
    position.eval()
    return detector, memory, decoder, position, vocabulary_size, checkpoint, config


def checkpoint_state(
    detector,
    memory,
    decoder,
    position,
    *,
    iteration: int,
    vocabulary_size: int,
    position_encoding: str,
) -> dict[str, Any]:
    """Create a checkpoint accepted by the packaged inference command."""

    return {
        "backbone_state": detector.img_backbone.state_dict(),
        "neck_state": detector.img_neck.state_dict(),
        "head_state": detector.head.state_dict(),
        "trajectory_modeling": memory.state_dict(),
        "id_decoder": decoder.state_dict(),
        "pos_enc": position.state_dict(),
        "pos_enc_mode": position_encoding,
        "iter": int(iteration),
        "vocab": int(vocabulary_size),
    }
