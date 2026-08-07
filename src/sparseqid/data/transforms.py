# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-time input pipeline for Sparse4D, ported from the TAO dataloader.

Only the transforms used at inference are kept (``ResizeCropFlipImage`` →
``NormalizeMultiviewImage`` → ``AICitySparse4DAdaptor``); semantics are
identical to upstream, including the PIL resize/crop path and the
BGR<->RGB channel swap inside ``normalize_image`` (the network effectively
sees BGR — see tao notes).
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def build_test_aug_config(image_size=(1080, 1920), final_dim=(540, 960)) -> Dict:
    """Mirror Omniverse3DDetTrackDataset.get_augmentation() test branch."""
    H, W = image_size
    fH, fW = final_dim
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    return {
        "resize": resize,
        "resize_dims": resize_dims,
        "crop": (crop_w, crop_h, crop_w + fW, crop_h + fH),
        "flip": False,
        "rotate": 0,
        "rotate_3d": 0,
        "frame_drop_prob": 0,
    }


def normalize_image(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std (in place, float32 input)."""
    assert img.dtype != np.uint8, f"img.dtype: {img.dtype} != np.uint8, Image is not uint8"
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


class ResizeCropFlipImage:
    """Resize, crop and flip images, updating ``lidar2img`` accordingly."""

    def __call__(self, results: Dict) -> Dict:
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results

        imgs = results["img"]
        new_imgs = []
        for i in range(len(imgs)):
            img, mat = self._img_transform(np.uint8(imgs[i]), aug_config)
            new_imgs.append(np.array(img).astype(np.float32))
            if "lidar2img" in results:
                results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img: np.ndarray, aug_configs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = aug_configs.get("resize_dims", (int(W * resize), int(H * resize)))
        crop = aug_configs.get("crop", [0, 0, resize_dims[0], resize_dims[1]])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]])
            transform_matrix = flip_matrix @ transform_matrix

        rotate_rad = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate_rad), np.sin(rotate_rad), 0],
                [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix

        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


class NormalizeMultiviewImage:
    """Normalize multi-view images."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results: Dict) -> Dict:
        if "img" not in results:
            return results
        results["img"] = [
            normalize_image(img, self.mean, self.std, self.to_rgb) for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class AICitySparse4DAdaptor:
    """Adapt data format for the Sparse4D model (test-time fields only)."""

    def __call__(self, results: Dict) -> Dict:
        if "lidar2img" in results:
            results["projection_mat"] = np.float32(np.stack(results["lidar2img"]))

        if "img_shape" in results:
            results["image_wh"] = np.ascontiguousarray(
                np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )

        if "cam_intrinsic" in results:
            results["cam_intrinsic"] = np.float32(np.stack(results["cam_intrinsic"]))
            results["focal"] = results["cam_intrinsic"][..., 0, 0]

        imgs = [img.transpose(2, 0, 1) for img in results["img"]]  # HWC -> CHW
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results["img"] = torch.tensor(imgs)
        return results
