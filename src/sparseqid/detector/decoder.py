# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder for Sparse4D."""

import torch

from .box3d import COS_YAW, H, L, SIN_YAW, VX, W, X, Y, Z


def decode_box(box):
    """Decode the box.

    Args:
        box (torch.Tensor): Box to decode.

    Returns:
        torch.Tensor: Decoded box.
    """
    yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
    box = torch.cat(
        [
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
            box[:, VX:],
        ],
        dim=-1,
    )
    return box
