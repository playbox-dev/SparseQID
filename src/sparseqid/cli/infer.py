# SPDX-License-Identifier: Apache-2.0

"""Run the complete SparseQID pipeline on one scene."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..build import load_pipeline
from ..constants import MODEL_IDX_TO_CLASS_ID
from ..data.calibration import load_calibration
from ..data.frame_dataset import FrameJpgDataset, batch_to_device, collate_scene
from ..detector.build import DEFAULT_CONFIG_DIR
from ..detector.decoder import decode_box
from ..submission import Track, write_tracks
from ..tracking import SparseQIDTracker


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog=prog, description=__doc__)
    result.add_argument("--checkpoint", "--ckpt", required=True)
    result.add_argument("--scene", required=True)
    result.add_argument("--split", default="val")
    result.add_argument("--frames", type=int, default=9000)
    result.add_argument(
        "--data-root", default=os.environ.get("AICITY26_DATA", "data/MTMC_Tracking_2026")
    )
    result.add_argument(
        "--cache-root", default=os.environ.get("AICITY26_CACHE", "data/aicity2026_frames_540")
    )
    result.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    result.add_argument("--output", default=None, help="11-column output; defaults by scene")
    result.add_argument("--scene-id", type=int, default=None)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--amp", action="store_true", help="use bfloat16 CUDA autocast")
    result.add_argument("--score-threshold", "--score", type=float, default=0.4)
    result.add_argument("--identity-threshold", type=float, default=0.2)
    result.add_argument("--newborn-threshold", type=float, default=0.6)
    result.add_argument("--spatial-gate", type=float, default=2.5)
    result.add_argument("--newborn-suppression", type=float, default=0.8)
    result.add_argument("--assignment", choices=("object-max", "hungarian"), default="object-max")
    result.add_argument(
        "--token-mode",
        choices=("combined", "appearance", "position"),
        default="combined",
        help="paper model uses combined; other modes reproduce token ablations",
    )
    return result


def _world_offset(data_root: Path, split: str, scene: str) -> np.ndarray:
    calibration = load_calibration(data_root / split / scene / "calibration.json")
    center = np.mean([camera.center for camera in calibration.cameras], axis=0)
    return np.asarray([-center[0], -center[1], 0.0])


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass --device cpu for a smoke run"
        )
    device = torch.device(args.device)
    detector, memory, decoder, position, vocabulary, checkpoint, _ = load_pipeline(
        args.checkpoint, device=device, config_dir=args.config_dir
    )
    tracker = SparseQIDTracker(
        decoder,
        memory,
        vocabulary,
        device,
        identity_threshold=args.identity_threshold,
        newborn_threshold=args.newborn_threshold,
        assignment=args.assignment,
        spatial_gate=args.spatial_gate,
        newborn_suppression=args.newborn_suppression,
    )
    data_root, cache_root = Path(args.data_root), Path(args.cache_root)
    dataset = FrameJpgDataset(
        cache_root,
        data_root,
        args.split,
        [args.scene],
        recenter=True,
        recenter_mode="camera",
        final_dim=(540, 960),
        max_frames=args.frames,
    )
    offset = _world_offset(data_root, args.split, args.scene)
    scene_id = args.scene_id
    if scene_id is None:
        digits = "".join(character for character in args.scene if character.isdigit())
        scene_id = int(digits or 0)
    output = Path(args.output or f"outputs/predictions/{args.scene}.txt")
    tracks: list[Track] = []
    scores: list[float] = []
    identities: set[int] = set()
    detector.head.return_feature = True
    detector.head.instance_bank.reset()

    for index, (_, frame_id) in enumerate(dataset.samples):
        batch = batch_to_device(collate_scene([dataset[index]]), device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"
            ),
        ):
            features = detector.extract_feat(batch["img"], batch)
            prediction = detector.head(features, batch)
        logits = prediction["classification"][-1][0]
        boxes = prediction["prediction"][-1][0]
        confidence, model_classes = logits.float().sigmoid().max(-1)
        selected = confidence >= args.score_threshold
        query_features = prediction["instance_feature"][0][selected].float()
        decoded = decode_box(boxes[selected])[:, :7].float().cpu()
        if args.token_mode == "position":
            query_features = position(boxes[selected, :3].float())
            position_features = None
        elif args.token_mode == "appearance":
            position_features = None
        else:
            position_features = position(boxes[selected, :3].float())
        global_ids = tracker.step(
            query_features,
            position_features=position_features,
            positions=decoded[:, :2],
            scores=confidence[selected].float().cpu(),
        )
        for detection, global_id in enumerate(global_ids.tolist()):
            if global_id < 0:
                continue
            class_id = MODEL_IDX_TO_CLASS_ID[int(model_classes[selected][detection])]
            x, y, z, width, length, height, yaw = decoded[detection].tolist()
            tracks.append(
                Track(
                    scene_id,
                    class_id,
                    global_id,
                    int(frame_id),
                    x - offset[0],
                    y - offset[1],
                    z - offset[2],
                    width,
                    length,
                    height,
                    yaw,
                )
            )
            scores.append(float(confidence[selected][detection]))
            identities.add(global_id)
        if index % 100 == 0:
            print(
                f"{args.scene}: {index + 1}/{len(dataset)} frames, {len(tracks)} rows", flush=True
            )

    write_tracks(tracks, output)
    output.with_suffix(output.suffix + ".scores").write_text(
        "".join(f"{score:.4f}\n" for score in scores)
    )
    summary = {
        "scene": args.scene,
        "frames": len(dataset),
        "rows": len(tracks),
        "track_ids": len(identities),
        "checkpoint_iteration": checkpoint.get("iter"),
        "vocabulary_size": vocabulary,
        "score_threshold": args.score_threshold,
        "token_mode": args.token_mode,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return output, summary


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    args = parser(prog).parse_args(argv)
    output, summary = run(args)
    print(f"wrote {summary['rows']} rows to {output} ({summary['track_ids']} track IDs)")


if __name__ == "__main__":
    main()
