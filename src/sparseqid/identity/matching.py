# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: Modified by the SparseQID authors.

"""Match frozen detector queries to identity labels during training."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..detector.box3d import YAW, H, L, W, X, Y, Z


class QueryMatcher:
    """Hungarian matcher for the identity-only training path."""

    def __init__(
        self,
        cls_weight: float = 2.0,
        box_weight: float = 0.25,
        reg_weights: list[float] | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
        matching_cost_threshold: float = 1e6,
    ) -> None:
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.reg_weights = reg_weights or ([1.0] * 8 + [0.0] * 2)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.matching_cost_threshold = matching_cost_threshold

    @staticmethod
    def _encode_boxes(boxes: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                boxes[..., [X, Y, Z]],
                boxes[..., [W, L, H]].log(),
                torch.sin(boxes[..., YAW]).unsqueeze(-1),
                torch.cos(boxes[..., YAW]).unsqueeze(-1),
                boxes[..., YAW + 1 :],
            ],
            dim=-1,
        )

    def __call__(
        self,
        class_logits: torch.Tensor,
        box_predictions: torch.Tensor,
        class_targets: list[torch.Tensor],
        box_targets: list[torch.Tensor],
        identity_targets: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return one identity target per query, or ``-1`` when unmatched."""
        batch_size, num_queries, _ = class_logits.shape
        matched = identity_targets[0].new_full((batch_size, num_queries), -1)
        probabilities = class_logits.sigmoid()
        regression_weights = box_predictions.new_tensor(self.reg_weights)

        for batch_index in range(batch_size):
            labels = class_targets[batch_index]
            if len(labels) == 0:
                continue

            probability = probabilities[batch_index]
            negative = (
                -(1 - probability + self.eps).log() * (1 - self.alpha) * probability.pow(self.gamma)
            )
            positive = (
                -(probability + self.eps).log() * self.alpha * (1 - probability).pow(self.gamma)
            )
            class_cost = (positive[:, labels] - negative[:, labels]) * self.cls_weight

            encoded_targets = self._encode_boxes(box_targets[batch_index]).to(
                box_predictions.device
            )
            finite_weights = torch.logical_not(encoded_targets.isnan()).to(encoded_targets.dtype)
            box_cost = (
                torch.sum(
                    torch.abs(box_predictions[batch_index, :, None] - encoded_targets[None])
                    * finite_weights[None]
                    * regression_weights,
                    dim=-1,
                )
                * self.box_weight
            )

            cost = (class_cost + box_cost).detach().cpu().numpy()
            cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
            if cost.size == 0 or np.min(cost) > self.matching_cost_threshold:
                continue
            query_indices, target_indices = linear_sum_assignment(cost)
            valid = cost[query_indices, target_indices] <= self.matching_cost_threshold
            query_indices = torch.as_tensor(
                query_indices[valid], device=matched.device, dtype=torch.long
            )
            target_indices = torch.as_tensor(
                target_indices[valid], device=matched.device, dtype=torch.long
            )
            matched[batch_index, query_indices] = identity_targets[batch_index][target_indices]

        return matched
