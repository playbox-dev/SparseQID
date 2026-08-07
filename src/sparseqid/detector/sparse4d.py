# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: Modified by the SparseQID authors.

"""Frozen Sparse4D feature extractor used by SparseQID."""

from typing import Dict

import torch.nn as nn

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck


class Sparse4D(nn.Module):
    """Recurrent outside-in detector architecture used by the paper checkpoints."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.model
        self.img_backbone = build_backbone(
            model_config.backbone.type,
            out_indices=[0, 1, 2, 3],
            freeze_norm=True,
        )
        del self.img_backbone.global_pool
        del self.img_backbone.fc
        self.img_backbone.set_grad_checkpointing(True)
        self.img_neck = build_neck(model_config.neck)
        self.head = build_head(config)

    def init_weights(self):
        """Initialize detector modules when constructing weights from scratch."""
        if hasattr(self.img_backbone, "init_weights"):
            self.img_backbone.init_weights()
        if self.img_neck is not None and hasattr(self.img_neck, "init_weights"):
            self.img_neck.init_weights()
        if hasattr(self.head, "init_weights"):
            self.head.init_weights()

    def extract_feat(self, img, metas=None):
        """Extract one feature pyramid from synchronized camera images."""
        batch_size = img.shape[0]
        if img.dim() == 5:
            num_cameras = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cameras = 1

        feature_maps = self.img_backbone.forward_feature_pyramid(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        return [
            feature.reshape((batch_size, num_cameras) + feature.shape[1:])
            for feature in feature_maps
        ]

    def forward(self, img, metas):
        """Return recurrent detector predictions for one synchronized frame."""
        return self.head(self.extract_feat(img, metas), metas)
