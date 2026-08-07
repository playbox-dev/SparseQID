# Portions Copyright (c) Ruopeng Gao. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for SparseQID.

"""The single online identity runtime used by SparseQID."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def _object_max_assignment(scores, active_labels, newborn_label, threshold):
    confidence, labels = scores.max(-1)
    best_for_label = {}
    for value, label in zip(confidence.tolist(), labels.tolist()):
        best_for_label[label] = max(best_for_label.get(label, 0.0), value)
    best_for_label[newborn_label] = 0.0

    assigned, used = [], set()
    for value, label in zip(confidence.tolist(), labels.tolist()):
        if (
            label not in active_labels
            or value < threshold
            or value < best_for_label[label]
            or label in used
        ):
            assigned.append(newborn_label)
        else:
            assigned.append(label)
            used.add(label)
    return torch.tensor(assigned, dtype=torch.long)


def _hungarian_assignment(scores, active_labels, newborn_label, threshold):
    detection_count = len(scores)
    padded = scores
    if detection_count > 1:
        padded = torch.cat([scores, scores[:, -1:].repeat(1, detection_count - 1)], -1)
    rows, columns = linear_sum_assignment((1 - padded).cpu().numpy())
    assigned = [newborn_label] * detection_count
    for row, column in zip(rows, columns):
        if (
            column < newborn_label
            and column in active_labels
            and float(padded[row, column]) >= threshold
        ):
            assigned[row] = int(column)
    return torch.tensor(assigned, dtype=torch.long)


class SparseQIDTracker:
    """Assign persistent scene IDs to one frame of sparse-query features at a time."""

    def __init__(
        self,
        identity_decoder,
        memory_encoder,
        vocabulary_size: int,
        device: str | torch.device,
        *,
        miss_tolerance: int = 30,
        identity_threshold: float = 0.2,
        newborn_threshold: float = 0.6,
        assignment: str = "object-max",
        spatial_gate: float | None = None,
        gate_growth: float = 0.15,
        newborn_suppression: float | None = None,
    ):
        self.identity_decoder = identity_decoder
        self.memory_encoder = memory_encoder
        self.vocabulary_size = int(vocabulary_size)
        self.device = torch.device(device)
        self.identity_threshold = float(identity_threshold)
        self.newborn_threshold = float(newborn_threshold)
        self.assignment = assignment
        self.spatial_gate = spatial_gate
        self.gate_growth = float(gate_growth)
        self.newborn_suppression = newborn_suppression
        self.window = max(
            1,
            min(int(miss_tolerance), int(identity_decoder.rel_pe_length)) - 1,
        )
        feature_dim = int(identity_decoder.feature_dim)
        self.features = torch.zeros((0, 0, feature_dim), device=self.device)
        self.masks = torch.zeros((0, 0), dtype=torch.bool, device=self.device)
        self.active_labels: list[int] = []
        self.reuse_queue = list(range(self.vocabulary_size))
        self.label_to_global_id: dict[int, int] = {}
        self.last_position: dict[int, tuple[int, tuple[float, float]]] = {}
        self.next_global_id = 0
        self.frame_index = 0

    def _touch(self, label: int) -> None:
        self.reuse_queue.remove(label)
        self.reuse_queue.append(label)

    def _predict_labels(self, features, positions):
        detection_count = len(features)
        newborn = self.vocabulary_size
        if detection_count == 0:
            return torch.zeros(0, dtype=torch.long)
        if not self.active_labels or self.features.shape[0] == 0:
            return torch.full((detection_count,), newborn, dtype=torch.long)

        time_count, track_count = self.features.shape[:2]
        times = torch.arange(time_count, dtype=torch.long, device=self.device)
        sequence = {
            "trajectory_features": self.features[None, None],
            "trajectory_id_labels": torch.tensor(
                self.active_labels, dtype=torch.long, device=self.device
            )
            .expand(time_count, track_count)[None, None]
            .contiguous(),
            "trajectory_times": times[:, None]
            .expand(time_count, track_count)[None, None]
            .contiguous(),
            "trajectory_masks": self.masks[None, None],
            "unknown_features": features[None, None, None],
            "unknown_masks": torch.zeros(
                (1, 1, 1, detection_count), dtype=torch.bool, device=self.device
            ),
            "unknown_times": torch.full(
                (1, 1, 1, detection_count),
                time_count,
                dtype=torch.long,
                device=self.device,
            ),
        }
        sequence = self.memory_encoder(sequence)
        logits, _, _ = self.identity_decoder(sequence, use_decoder_checkpoint=False)
        scores = logits[0, 0, 0].float().softmax(-1)

        if self.spatial_gate is not None and positions is not None:
            current = torch.as_tensor(positions, dtype=torch.float32, device=self.device)[:, :2]
            for label in self.active_labels:
                if label not in self.last_position:
                    continue
                previous_frame, previous_xy = self.last_position[label]
                radius = self.spatial_gate + self.gate_growth * max(
                    self.frame_index - previous_frame, 1
                )
                previous = torch.tensor(previous_xy, dtype=torch.float32, device=self.device)
                scores[(current - previous).norm(dim=-1) > radius, label] = 0.0

        active = set(self.active_labels)
        if self.assignment == "object-max":
            return _object_max_assignment(scores, active, newborn, self.identity_threshold)
        if self.assignment == "hungarian":
            return _hungarian_assignment(scores, active, newborn, self.identity_threshold)
        raise ValueError(f"unknown assignment protocol {self.assignment!r}")

    @torch.no_grad()
    def step(self, features, *, position_features=None, positions=None, scores=None):
        """Advance one frame and return a CPU tensor of global IDs; ``-1`` means dropped."""

        features = features.to(self.device)
        if position_features is not None and len(features):
            features = features + position_features.to(self.device)
        detection_count = len(features)
        newborn = self.vocabulary_size
        labels = self._predict_labels(features, positions)

        keep = torch.ones(detection_count, dtype=torch.bool)
        if scores is not None and detection_count:
            detection_scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
            keep = (labels != newborn) | (detection_scores >= self.newborn_threshold)

        if self.newborn_suppression is not None and positions is not None and detection_count:
            claimed = set(labels[keep & (labels != newborn)].tolist())
            current = torch.as_tensor(positions, dtype=torch.float32)[:, :2]
            candidates = []
            for detection in torch.nonzero(keep & (labels == newborn)).flatten().tolist():
                for label in self.active_labels:
                    if label in claimed or label not in self.last_position:
                        continue
                    previous_frame, previous_xy = self.last_position[label]
                    distance = float((current[detection] - torch.tensor(previous_xy)).norm())
                    radius = self.newborn_suppression + self.gate_growth * max(
                        self.frame_index - previous_frame, 1
                    )
                    if distance <= radius:
                        candidates.append((distance, detection, label))
            used_detections, used_labels = set(), set()
            for _, detection, label in sorted(candidates):
                if detection in used_detections or label in used_labels:
                    continue
                labels[detection] = label
                used_detections.add(detection)
                used_labels.add(label)

        for label in labels[keep & (labels != newborn)].tolist():
            self._touch(int(label))

        remaining_slots = self.vocabulary_size - int((keep & (labels != newborn)).sum())
        newborn_detections = torch.nonzero(keep & (labels == newborn)).flatten().tolist()
        for detection in newborn_detections[max(remaining_slots, 0) :]:
            keep[detection] = False

        assigned = labels.clone()
        active_now = set(assigned[keep & (assigned != newborn)].tolist())
        newborn_detections = torch.nonzero(keep & (labels == newborn)).flatten().tolist()
        reusable = [label for label in self.reuse_queue if label not in active_now][
            : len(newborn_detections)
        ]
        for detection, label in zip(newborn_detections, reusable):
            if label in self.active_labels:
                column = self.active_labels.index(label)
                selected = [index for index in range(len(self.active_labels)) if index != column]
                self.features = self.features[:, selected]
                self.masks = self.masks[:, selected]
                self.active_labels.pop(column)
            self.last_position.pop(label, None)
            self.label_to_global_id[label] = self.next_global_id
            self.next_global_id += 1
            assigned[detection] = label
            self._touch(label)

        if self.window > 1:
            self.features = self.features[-(self.window - 1) :]
            self.masks = self.masks[-(self.window - 1) :]
        else:
            self.features = self.features[:0]
            self.masks = self.masks[:0]

        feature_dim = int(self.identity_decoder.feature_dim)
        for detection in torch.nonzero(keep).flatten().tolist():
            label = int(assigned[detection])
            if label not in self.active_labels:
                self.active_labels.append(label)
                self.features = torch.cat(
                    [
                        self.features,
                        torch.zeros((self.features.shape[0], 1, feature_dim), device=self.device),
                    ],
                    1,
                )
                self.masks = torch.cat(
                    [
                        self.masks,
                        torch.ones((self.masks.shape[0], 1), dtype=torch.bool, device=self.device),
                    ],
                    1,
                )

        row_features = torch.zeros((1, len(self.active_labels), feature_dim), device=self.device)
        row_masks = torch.ones((1, len(self.active_labels)), dtype=torch.bool, device=self.device)
        for detection in torch.nonzero(keep).flatten().tolist():
            column = self.active_labels.index(int(assigned[detection]))
            row_features[0, column] = features[detection]
            row_masks[0, column] = False
        self.features = torch.cat([self.features, row_features], 0)
        self.masks = torch.cat([self.masks, row_masks], 0)

        alive = (~self.masks).any(0)
        if not bool(alive.all()):
            selected = torch.nonzero(alive).flatten().tolist()
            self.features = self.features[:, selected]
            self.masks = self.masks[:, selected]
            self.active_labels = [self.active_labels[index] for index in selected]

        global_ids = torch.full((detection_count,), -1, dtype=torch.long)
        for detection in torch.nonzero(keep).flatten().tolist():
            label = int(assigned[detection])
            global_ids[detection] = self.label_to_global_id[label]
            if positions is not None:
                xy = positions[detection]
                self.last_position[label] = (
                    self.frame_index,
                    (float(xy[0]), float(xy[1])),
                )
        self.frame_index += 1
        return global_ids


def track_clip(
    identity_decoder,
    memory_encoder,
    frame_features,
    vocabulary_size,
    device,
    *,
    position_features=None,
    frame_scores=None,
    frame_positions=None,
    **tracker_options,
):
    """Run ``SparseQIDTracker`` over a finite clip; useful for training tests."""

    tracker = SparseQIDTracker(
        identity_decoder,
        memory_encoder,
        vocabulary_size,
        device,
        **tracker_options,
    )
    output = []
    for index, features in enumerate(frame_features):
        output.append(
            tracker.step(
                features,
                position_features=(None if position_features is None else position_features[index]),
                positions=None if frame_positions is None else frame_positions[index],
                scores=None if frame_scores is None else frame_scores[index],
            )
        )
    return output
