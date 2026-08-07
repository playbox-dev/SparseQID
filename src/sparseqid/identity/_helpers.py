# Portions Copyright (c) Ruopeng Gao. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for SparseQID.

"""Small identity-model helpers adapted from MOTIP."""

import copy

import numpy as np
import torch
import torch.nn as nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def label_to_one_hot(labels: torch.Tensor, n_classes: int, dtype=torch.float32):
    return torch.eye(n=n_classes, device=labels.device, dtype=dtype)[labels]


def labels_to_one_hot(labels: np.ndarray, class_num: int):  # only used for focal-loss path
    labels = labels.cpu()
    return np.eye(N=class_num)[labels].reshape((len(labels), -1))


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def distributed_world_size():
    return torch.distributed.get_world_size() if is_distributed() else 1
