# SPDX-License-Identifier: Apache-2.0

"""Train the SparseQID identity model on a frozen recurrent detector."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as distributed

from ..build import build_identity, checkpoint_state, load_base_detector_state
from ..data.frame_dataset import FrameJpgDataset, batch_to_device, collate_scene
from ..detector.build import DEFAULT_CONFIG_DIR, build_architecture
from ..identity import IDCriterion, QueryMatcher, assemble_seq_info, build_id_targets

PAPER_SCENES = [f"Warehouse_{index:03d}" for index in range(20)] + [
    f"2025:Warehouse_{index:03d}" for index in range(15)
]


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog=prog, description=__doc__)
    result.add_argument(
        "--base-checkpoint", required=True, help="NGC or SparseQID detector checkpoint"
    )
    result.add_argument("--output", default="checkpoints/sparseqid.pth")
    result.add_argument("--scenes", nargs="+", default=PAPER_SCENES)
    result.add_argument("--split", default="train")
    result.add_argument(
        "--data-root", default=os.environ.get("AICITY26_DATA", "data/MTMC_Tracking_2026")
    )
    result.add_argument(
        "--cache-root", default=os.environ.get("AICITY26_CACHE", "data/aicity2026_frames_540")
    )
    result.add_argument(
        "--data-root-2025", default=os.environ.get("AICITY25_DATA", "data/MTMC_Tracking_2025")
    )
    result.add_argument(
        "--cache-root-2025", default=os.environ.get("AICITY25_CACHE", "data/aicity2025_frames_540")
    )
    result.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    result.add_argument("--iterations", type=int, default=6000)
    result.add_argument("--checkpoint-every", type=int, default=500)
    result.add_argument("--clip-length", type=int, default=30)
    result.add_argument("--interval-min", type=int, default=1)
    result.add_argument("--interval-max", type=int, default=4)
    result.add_argument("--augmentation-groups", type=int, default=6)
    result.add_argument("--vocabulary-size", type=int, default=128)
    result.add_argument("--occlusion-probability", type=float, default=0.5)
    result.add_argument("--switch-probability", type=float, default=0.5)
    result.add_argument("--learning-rate", type=float, default=4e-4)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--position-encoding", choices=("fourier", "raw", "mlp"), default="fourier")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default=None)
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="print the resolved recipe and exit")
    return result


class _Scene:
    def __init__(self, name: str, args: argparse.Namespace):
        data_root, cache_root, scene = args.data_root, args.cache_root, name
        if name.startswith("2025:"):
            scene = name.split(":", 1)[1]
            data_root, cache_root = args.data_root_2025, args.cache_root_2025
        self.dataset = FrameJpgDataset(
            cache_root,
            data_root,
            args.split,
            [scene],
            recenter=True,
            recenter_mode="camera",
            final_dim=(540, 960),
        )
        self.index_by_frame = {
            frame_id: index for index, (_, frame_id) in enumerate(self.dataset.samples)
        }
        self.frames = sorted(self.index_by_frame)

    def sample(self, frame_id: int):
        return self.dataset[self.index_by_frame[frame_id]]


def _sample_frames(frames: list[int], length: int, interval: int) -> list[int]:
    span = (length - 1) * interval
    if len(frames) <= span:
        return frames[:length]
    start = random.randint(0, len(frames) - 1 - span)
    return [frames[start + step * interval] for step in range(length)]


def _query_matcher(config) -> QueryMatcher:
    values = config.model.head.matching
    return QueryMatcher(
        cls_weight=values.cls_weight,
        box_weight=values.box_weight,
        reg_weights=values.reg_weights,
    )


def _matched_queries(matcher, prediction, batch):
    classes = prediction["classification"][-1].detach().float()
    boxes = prediction["prediction"][-1].detach().float()
    identities = matcher(
        classes,
        boxes,
        batch["gt_labels_3d"],
        batch["gt_bboxes_3d"],
        batch["instance_id"],
    )
    selected = identities[0] >= 0
    return (
        prediction["instance_feature"][0][selected].detach().float(),
        identities[0][selected].detach().cpu(),
        boxes[0][selected, :3].detach().float(),
    )


def _distributed_setup(args):
    if "RANK" not in os.environ:
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, torch.device(device)
    local_rank = int(os.environ["LOCAL_RANK"])
    distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return (
        True,
        distributed.get_rank(),
        distributed.get_world_size(),
        torch.device(f"cuda:{local_rank}"),
    )


def _average_gradients(parameters, world_size: int) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _save(path, detector, memory, decoder, position, args, iteration):
    state = checkpoint_state(
        detector,
        memory,
        decoder,
        position,
        iteration=iteration,
        vocabulary_size=args.vocabulary_size,
        position_encoding=args.position_encoding,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)


def run(args: argparse.Namespace) -> None:
    distributed_run, rank, world_size, device = _distributed_setup(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed + rank)
    detector, config = build_architecture(args.config_dir, device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    load_base_detector_state(detector, base)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    detector.eval()
    detector.head.return_feature = True
    matcher = _query_matcher(config)
    memory, decoder, position, _ = build_identity(
        vocabulary_size=args.vocabulary_size,
        decoder_layers=6,
        memory_length=args.clip_length,
        position_encoding=args.position_encoding,
        device=device,
    )
    identity_loss = IDCriterion(weight=1.0, use_focal_loss=False)
    trainable = list(memory.parameters()) + list(decoder.parameters()) + list(position.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-3)
    scenes = {name: _Scene(name, args) for name in args.scenes}
    if rank == 0:
        print(f"training on {len(scenes)} scenes with {world_size} process(es) on {device}")

    memory.train()
    decoder.train()
    position.train()
    running_loss = 0.0
    completed = 0
    for iteration in range(args.iterations):
        scene = scenes[random.choice(args.scenes)]
        interval = random.randint(args.interval_min, args.interval_max)
        frame_ids = _sample_frames(scene.frames, args.clip_length, interval)
        detector.head.instance_bank.reset()
        frame_features, track_ids, position_features = [], [], []
        for frame_id in frame_ids:
            batch = batch_to_device(collate_scene([scene.sample(frame_id)]), device)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=not args.no_amp and device.type == "cuda",
                ),
            ):
                feature_maps = detector.extract_feat(batch["img"], batch)
                prediction = detector.head(feature_maps, batch)
            features, identities, centers = _matched_queries(matcher, prediction, batch)
            frame_features.append(features)
            track_ids.append(identities)
            position_features.append(position(centers))
        if not any(len(features) for features in frame_features):
            continue
        targets = build_id_targets(
            track_ids,
            args.vocabulary_size,
            args.augmentation_groups,
            args.vocabulary_size,
            args.occlusion_probability,
            args.switch_probability,
        )
        sequence = assemble_seq_info(targets, frame_features, position_features)
        sequence = memory(sequence)
        logits, labels, masks = decoder(sequence, use_decoder_checkpoint=False)
        loss = identity_loss(logits, labels, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if distributed_run:
            _average_gradients(trainable, world_size)
        torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        running_loss += float(loss.detach())
        completed += 1
        if rank == 0 and iteration % 25 == 0:
            print(
                f"iteration {iteration:5d}  identity_loss={running_loss / completed:.4f}",
                flush=True,
            )
            running_loss, completed = 0.0, 0
        if rank == 0 and args.checkpoint_every and (iteration + 1) % args.checkpoint_every == 0:
            _save(args.output, detector, memory, decoder, position, args, iteration + 1)
    if distributed_run:
        distributed.barrier()
    if rank == 0:
        _save(args.output, detector, memory, decoder, position, args, args.iterations)
        print(f"saved {args.output}")
    if distributed_run:
        distributed.destroy_process_group()


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    args = parser(prog).parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
