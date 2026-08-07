"""Load synchronized multi-camera JPEG frames and optional 3D ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .calibration import load_calibration
from .transforms import (
    AICitySparse4DAdaptor,
    NormalizeMultiviewImage,
    ResizeCropFlipImage,
    build_test_aug_config,
)

FPS = 30.0
TYPE_TO_LABEL = {
    "Person": 0,
    "FourierGR1T2": 1,
    "AgilityDigit": 2,
    "NovaCarter": 3,
    "Transporter": 4,
    "Forklift": 5,
    "PalletTruck": 6,
}


def _fill_gt_velocity(gtc, fps):
    """Derive per-object ground-plane velocity (vx,vy) by central difference of
    position across consecutive GT frames (the json is dense at all frames) and
    write it into box cols 7,8 (vz left 0). Matches TAO's gt_velocity signal, which
    the json doesn't ship. Translation-invariant, so recenter doesn't matter."""
    track = {}  # id -> {frame: (row_idx, x, y)}
    for fr, (bb, ll, ii) in gtc.items():
        for k, tid in enumerate(ii.tolist()):
            track.setdefault(tid, {})[fr] = (k, float(bb[k, 0]), float(bb[k, 1]))
    for tid, seq in track.items():
        frs = sorted(seq)
        for i, fr in enumerate(frs):
            fp, fn = frs[max(0, i - 1)], frs[min(len(frs) - 1, i + 1)]
            dt = (fn - fp) / fps
            if dt <= 0:
                continue
            _, xp, yp = seq[fp]
            _, xn, yn = seq[fn]
            k = seq[fr][0]
            gtc[fr][0][k, 7] = (xn - xp) / dt
            gtc[fr][0][k, 8] = (yn - yp) / dt


class FrameJpgDataset(Dataset):
    """One sample per synchronized frame, using the submitted detector preprocessing."""

    def __init__(
        self,
        cache_root,
        data_root,
        split,
        scenes,
        recenter=False,
        final_dim=(540, 960),
        recenter_mode="gt",
        max_frames=None,
    ):
        self.cache_root = Path(cache_root) / split
        self.data_root = Path(data_root) / split
        self.recenter = recenter
        self.recenter_mode = recenter_mode
        self.scenes = list(scenes)
        self.final_dim = tuple(final_dim)
        self.samples: list[tuple[str, int]] = []
        for sc in self.scenes:
            scd = self.cache_root / sc
            if not scd.is_dir():
                continue
            cam_dirs = sorted(d.name for d in scd.iterdir() if d.is_dir())
            if not cam_dirs:
                continue
            fis = sorted(int(p.stem) for p in (scd / cam_dirs[0]).glob("*.jpg"))
            if max_frames is not None:
                fis = fis[:max_frames]
            self.samples.extend((sc, fi) for fi in fis)
        if not self.samples:
            raise RuntimeError(
                f"no extracted frames found under {self.cache_root} for {self.scenes}"
            )
        self._cache = {}
        first_image = cv2.imread(str(next(self.cache_root.glob("*/*/*.jpg"))))
        self.image_size = first_image.shape[:2]

    def __len__(self):
        return len(self.samples)

    def _scene(self, sc):
        s = self._cache.get(sc)
        if s is None:
            calib = load_calibration(self.data_root / sc / "calibration.json")
            cams = calib.cameras
            base_proj = []
            for cam in cams:
                m = np.eye(4, dtype=np.float64)
                m[:3, :4] = cam.P
                m[0] *= self.image_size[1] / cam.width
                m[1] *= self.image_size[0] / cam.height
                base_proj.append(m)
            gt_path = self.data_root / sc / "ground_truth.json"
            gtc = {}
            if gt_path.exists():
                with gt_path.open() as handle:
                    raw = json.load(handle)
                for f, objs in raw.items():
                    b, lab, ids = [], [], []
                    for o in objs:
                        c = TYPE_TO_LABEL.get(o["object type"])
                        if c is None:
                            continue
                        x, y, z = o["3d location"]
                        w, l, h = o["3d bounding box scale"]
                        b.append(
                            [x, y, z, w, l, h, o["3d bounding box rotation"][2], 0.0, 0.0, 0.0]
                        )
                        lab.append(c)
                        ids.append(int(o["object id"]))
                    gtc[int(f)] = (
                        np.array(b, np.float32).reshape(-1, 10),
                        np.array(lab, np.int64),
                        np.array(ids, np.int64),
                    )
            elif self.recenter and self.recenter_mode == "gt":
                raise FileNotFoundError(
                    f"{sc}: recenter_mode='gt' needs ground_truth.json (absent here). "
                    f"Use recenter_mode='camera' for GT-free (test) inference."
                )
            if self.recenter:
                if self.recenter_mode == "camera":
                    c = np.mean([cam.center for cam in cams], axis=0)
                    t = np.array([-c[0], -c[1], 0.0])
                else:
                    allxy = np.concatenate([bx[:, :2] for bx, _, _ in gtc.values() if len(bx)])
                    t = np.array([-allxy[:, 0].mean(), -allxy[:, 1].mean(), 0.0])
                Tneg = np.eye(4, dtype=np.float64)
                Tneg[:3, 3] = -t
                base_proj = [m @ Tneg for m in base_proj]
                for fi in gtc:
                    bx, lb, ic = gtc[fi]
                    if len(bx):
                        bx[:, :3] += t
            _fill_gt_velocity(gtc, FPS)  # derive vx,vy (TAO's gt_velocity); json has none
            s = {
                "cam_ids": calib.camera_ids,
                "base_proj": base_proj,
                "gt": gtc,
                "aug": build_test_aug_config(image_size=self.image_size, final_dim=self.final_dim),
                "resize": ResizeCropFlipImage(),
                "norm": NormalizeMultiviewImage(
                    [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], to_rgb=True
                ),
                "adapt": AICitySparse4DAdaptor(),
            }
            self._cache[sc] = s
        return s

    def _gt_boxes(self, gt, fi):
        b, lab, ids = gt.get(fi, None) or (
            np.zeros((0, 10), np.float32),
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
        )
        return b, lab, ids

    def __getitem__(self, i):
        sc, fi = self.samples[i]
        s = self._scene(sc)
        imgs = []
        for c in s["cam_ids"]:
            bgr = cv2.imread(str(self.cache_root / sc / c / f"{fi:06d}.jpg"))
            if bgr is None:
                raise FileNotFoundError(f"missing cached frame {sc}/{c}/{fi:06d}.jpg")
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32))
        boxes, labels, ids = self._gt_boxes(s["gt"], fi)
        res = {
            "img": imgs,
            "lidar2img": [m.copy() for m in s["base_proj"]],
            "img_shape": [im.shape for im in imgs],
            "aug_config": s["aug"],
            "gt_bboxes_3d": boxes.copy(),
        }
        res = s["resize"](res)
        res = s["adapt"](s["norm"](res))
        return {
            "img": res["img"],  # (N, 3, fH, fW)
            "projection_mat": torch.as_tensor(res["projection_mat"]),
            "image_wh": torch.as_tensor(res["image_wh"]),
            "timestamp": torch.tensor([fi / FPS], dtype=torch.float64),
            "gt_bboxes_3d": torch.as_tensor(res["gt_bboxes_3d"], dtype=torch.float32),
            "gt_labels_3d": torch.as_tensor(labels, dtype=torch.long),
            "instance_id": torch.as_tensor(ids, dtype=torch.long),
        }


def collate_scene(samples):
    """Stack a same-scene batch. img/proj/wh stack (same #cams); GT stays a list."""
    return {
        "img": torch.stack([s["img"] for s in samples]),  # [B,N,3,fH,fW]
        "projection_mat": torch.stack([s["projection_mat"] for s in samples]),
        "image_wh": torch.stack([s["image_wh"] for s in samples]),
        "timestamp": torch.cat([s["timestamp"] for s in samples]),  # [B]
        "gt_bboxes_3d": [s["gt_bboxes_3d"] for s in samples],
        "gt_labels_3d": [s["gt_labels_3d"] for s in samples],
        "instance_id": [s["instance_id"] for s in samples],
    }


def batch_to_device(batch, device):
    """Move a collated frame batch to ``device``."""
    ids = [t.to(device) for t in batch["instance_id"]]
    return {
        "img": batch["img"].to(device, non_blocking=True),
        "projection_mat": batch["projection_mat"].to(device, non_blocking=True),
        "image_wh": batch["image_wh"].to(device, non_blocking=True),
        "timestamp": batch["timestamp"].to(device, non_blocking=True),
        "gt_bboxes_3d": [t.to(device) for t in batch["gt_bboxes_3d"]],
        "gt_labels_3d": [t.to(device) for t in batch["gt_labels_3d"]],
        "instance_id": ids,
    }
