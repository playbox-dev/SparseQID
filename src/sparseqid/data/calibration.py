"""Calibration loading and world->image projection.

The dataset's ``calibration.json`` provides, per camera sensor:
  - ``intrinsicMatrix``  : 3x3 K
  - ``extrinsicMatrix``  : 3x4 world->camera [R|t]
  - ``cameraMatrix``     : 3x4 full world->image projection P  (== K @ [R|t], but we
                           use it directly per the verified data contract)
  - ``homography``       : 3x3 ground-plane homography
  - ``attributes``       : fps, direction3d, frameWidth, frameHeight

Verified projection convention (sticky-note rule #1):
    uvw = P @ [x, y, z, 1]^T ;  (u, v) = uvw[:2] / uvw[2] ;  w = uvw[2] > 0  (in front)
Do NOT recompose P from K @ [R|t]; use ``cameraMatrix`` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """A single calibrated camera in a scene."""

    id: str
    K: np.ndarray  # (3, 3)
    extrinsic: np.ndarray  # (3, 4) world->cam
    P: np.ndarray  # (3, 4) world->image projection (cameraMatrix)
    homography: np.ndarray  # (3, 3)
    width: int
    height: int
    fps: float

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T t (from extrinsic [R|t])."""
        R = self.extrinsic[:, :3]
        t = self.extrinsic[:, 3]
        return -R.T @ t

    @property
    def forward(self) -> np.ndarray:
        """Camera optical-axis (+Z of camera frame) expressed in world coordinates."""
        R = self.extrinsic[:, :3]
        return R.T @ np.array([0.0, 0.0, 1.0])

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project Nx3 world points to image.

        Returns:
            uv:    (N, 2) pixel coordinates.
            valid: (N,) bool, True where the point is in front of the camera (w > 0).
        """
        pts = np.asarray(points_world, dtype=np.float64)
        squeeze = pts.ndim == 1
        if squeeze:
            pts = pts[None]
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
        uvw = homog @ self.P.T  # (N, 3)
        w = uvw[:, 2]
        valid = w > 1e-6
        safe_w = np.where(valid, w, 1.0)
        uv = uvw[:, :2] / safe_w[:, None]
        if squeeze:
            return uv[0], valid[0]
        return uv, valid


@dataclass
class SceneCalibration:
    """All cameras of a scene, in a stable sorted order by camera id."""

    cameras: list[Camera]

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def camera_ids(self) -> list[str]:
        return [c.id for c in self.cameras]

    def by_id(self, cam_id: str) -> Camera:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(cam_id)

    @property
    def projection_matrices(self) -> np.ndarray:
        """Stacked (num_cameras, 3, 4) projection matrices in ``cameras`` order."""
        return np.stack([c.P for c in self.cameras], axis=0)


def _attr(sensor: dict, name: str) -> str | None:
    for a in sensor.get("attributes", []):
        if a.get("name") == name:
            return a.get("value")
    return None


def load_calibration(path: str | Path) -> SceneCalibration:
    """Load ``calibration.json`` into a :class:`SceneCalibration`."""
    with open(path) as f:
        data = json.load(f)

    cameras: list[Camera] = []
    for sensor in data["sensors"]:
        if sensor.get("type") != "camera":
            continue
        width = int(_attr(sensor, "frameWidth") or 1920)
        height = int(_attr(sensor, "frameHeight") or 1080)
        fps = float(_attr(sensor, "fps") or 30.0)
        K = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64)
        E = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64)
        # Recompose P = K @ [R|t] rather than using the stored ``cameraMatrix``.
        # In the 2026 calib ~36% of cameras have a cameraMatrix with NEGATIVE
        # homogeneous w for in-view points (correct pixel, but w<0), which the
        # w>0 validity check (here and in the model's deformable sampling) rejects
        # -> those cameras silently contribute nothing. K@[R|t] gives identical
        # pixels on the good cameras and consistent positive-w on all of them.
        P = K @ E[:3, :4]
        cameras.append(
            Camera(
                id=sensor["id"],
                K=K,
                extrinsic=E,
                P=P,
                homography=np.asarray(sensor["homography"], dtype=np.float64),
                width=width,
                height=height,
                fps=fps,
            )
        )

    cameras.sort(key=lambda c: c.id)
    return SceneCalibration(cameras=cameras)
