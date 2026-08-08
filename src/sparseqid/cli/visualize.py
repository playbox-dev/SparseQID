# SPDX-License-Identifier: Apache-2.0

"""Render tracked 3D boxes from a predictions file as MP4 clips."""

from __future__ import annotations

import argparse
import colorsys
import os
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ..constants import CLASS_ID_TO_NAME
from ..data.calibration import Camera, SceneCalibration, load_calibration
from ..data.frame_dataset import FPS
from ..geometry import BOTTOM_FACE, BOX_EDGES, boxes_to_corners

TRAIL_LENGTH = 60


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog=prog, description=__doc__)
    result.add_argument("--preds", required=True, help="11-column file written by sqid infer")
    result.add_argument("--scene", required=True)
    result.add_argument("--split", default="val")
    result.add_argument(
        "--data-root", default=os.environ.get("AICITY26_DATA", "data/MTMC_Tracking_2026")
    )
    result.add_argument(
        "--cache-root", default=os.environ.get("AICITY26_CACHE", "data/aicity2026_frames_540")
    )
    result.add_argument("--out", default="outputs/viz", help="output directory")
    result.add_argument(
        "--frames",
        default="0:900",
        help="source frame range START:END or START:END:STEP; END is exclusive",
    )
    result.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="camera ids to render; defaults to the first camera of the scene",
    )
    result.add_argument("--view", choices=("camera", "bev", "both"), default="both")
    result.add_argument(
        "--color-by",
        choices=("id", "class"),
        default="id",
        help="id gives every track its own colour, which makes identity switches visible",
    )
    result.add_argument("--fps", type=float, default=None, help="default: real-time playback")
    result.add_argument("--bev-size", type=int, default=900, help="BEV image size in pixels")
    return result


def parse_frame_range(text: str) -> range:
    """Parse ``START:END`` or ``START:END:STEP`` into a range."""
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected START:END or START:END:STEP, got {text!r}")
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"frame range {text!r} must contain integers") from error
    start, end = values[0], values[1]
    step = values[2] if len(parts) == 3 else 1
    if step < 1:
        raise ValueError(f"frame range step must be positive, got {step}")
    if end <= start:
        raise ValueError(f"frame range end must exceed start, got {text!r}")
    return range(start, end, step)


def read_predictions(path: str | Path) -> dict[int, list[tuple[int, int, np.ndarray]]]:
    """Read an 11-column file into ``frame_id -> [(class_id, object_id, box7)]``."""
    per_frame: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) != 11:
                raise ValueError(
                    f"{path}:{number}: expected 11 columns, got {len(columns)}. "
                    "This command reads the output of sqid infer."
                )
            _, class_id, object_id, frame_id = (int(float(value)) for value in columns[:4])
            box = np.array([float(value) for value in columns[4:]], dtype=np.float64)
            per_frame[frame_id].append((class_id, object_id, box))
    return dict(per_frame)


def color_for(class_id: int, object_id: int, mode: str) -> tuple[int, int, int]:
    """Return a stable BGR colour for a detection."""
    key = class_id if mode == "class" else object_id
    # Golden-ratio hue stepping keeps neighbouring keys visually far apart.
    hue = (key * 0.618033988749895) % 1.0
    blue, green, red = colorsys.hsv_to_rgb(hue, 0.85, 1.0)[::-1]
    return int(blue * 255), int(green * 255), int(red * 255)


def _label(class_id: int, object_id: int) -> str:
    return f"{CLASS_ID_TO_NAME.get(class_id, class_id)}:{object_id}"


def _draw_label(image, text: str, anchor, color) -> None:
    position = (int(anchor[0]), int(anchor[1]) - 6)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_camera_view(image, camera: Camera, detections, color_by: str) -> int:
    """Draw projected boxes onto a cached frame; returns how many were visible."""
    height, width = image.shape[:2]
    # The cache holds resized frames, so scale the projection to the cached size.
    scale = np.array([width / camera.width, height / camera.height])
    drawn = 0
    for class_id, object_id, box in detections:
        corners = boxes_to_corners(box[:7])[0]
        uv, valid = camera.project(corners)
        if not valid.all():
            continue  # at least one corner is behind the camera
        uv = uv * scale
        on_image = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if not on_image.any():
            continue
        color = color_for(class_id, object_id, color_by)
        points = uv.round().astype(int)
        for start, end in BOX_EDGES:
            cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
        _draw_label(image, _label(class_id, object_id), points[on_image.argmax()], color)
        drawn += 1
    return drawn


class BirdsEyeView:
    """Fixed-extent top-down renderer with a fading trail per track."""

    def __init__(self, calibration: SceneCalibration, boxes: np.ndarray, size: int, margin=6.0):
        points = [np.asarray([camera.center[:2] for camera in calibration.cameras])]
        if len(boxes):
            points.append(boxes[:, :2])
        stacked = np.concatenate(points, axis=0)
        self.lower = stacked.min(axis=0) - margin
        upper = stacked.max(axis=0) + margin
        extent = np.maximum(upper - self.lower, 1e-6)
        # One scale for both axes keeps the view metrically square.
        self.scale = (size - 1) / extent.max()
        self.width = int(round(extent[0] * self.scale)) + 1
        self.height = int(round(extent[1] * self.scale)) + 1
        self.cameras = np.asarray([camera.center[:2] for camera in calibration.cameras])
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    def to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """World metres to image pixels, flipping y so north points up."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        columns = (xy[:, 0] - self.lower[0]) * self.scale
        rows = self.height - 1 - (xy[:, 1] - self.lower[1]) * self.scale
        return np.stack([columns, rows], axis=-1)

    def render(self, detections, color_by: str, frame_id: int) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        for center in self.to_pixels(self.cameras).round().astype(int):
            cv2.drawMarker(image, tuple(center), (90, 90, 90), cv2.MARKER_TRIANGLE_UP, 9, 1)

        present = set()
        for class_id, object_id, box in detections:
            present.add(object_id)
            self.trails[object_id].append(tuple(box[:2]))
            color = color_for(class_id, object_id, color_by)
            footprint = boxes_to_corners(box[:7])[0][list(BOTTOM_FACE)][:, :2]
            polygon = self.to_pixels(footprint).round().astype(np.int32)
            cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
            _draw_label(image, str(object_id), polygon.min(axis=0), color)
        for object_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            points = self.to_pixels(np.asarray(trail)).round().astype(np.int32)
            faded = color_for(0, object_id, "id")
            cv2.polylines(image, [points], False, tuple(value // 2 for value in faded), 1)
        for object_id in list(self.trails):
            if object_id not in present:  # let a departed track's trail decay away
                self.trails[object_id].append(self.trails[object_id][-1])

        cv2.putText(
            image,
            f"frame {frame_id}  tracks {len(present)}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return image


def _open_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open an MP4 writer for {path}")
    return writer


def _cached_frames(camera_dir: Path, wanted: range) -> list[int]:
    available = sorted(int(path.stem) for path in camera_dir.glob("*.jpg"))
    if not available:
        raise FileNotFoundError(f"no cached frames in {camera_dir}; run sqid extract first")
    selected = [frame for frame in available if frame in wanted]
    if not selected:
        raise ValueError(
            f"no cached frames in range {wanted.start}:{wanted.stop}:{wanted.step}; "
            f"{camera_dir.name} holds {available[0]}..{available[-1]}"
        )
    return selected


def run(args: argparse.Namespace) -> list[Path]:
    wanted = parse_frame_range(args.frames)
    scene_dir = Path(args.data_root) / args.split / args.scene
    calibration = load_calibration(scene_dir / "calibration.json")
    scene_cache = Path(args.cache_root) / args.split / args.scene
    camera_ids = args.cameras or calibration.camera_ids[:1]
    unknown = [camera for camera in camera_ids if camera not in calibration.camera_ids]
    if unknown:
        raise ValueError(f"unknown camera ids {unknown}; scene has {calibration.camera_ids}")

    predictions = read_predictions(args.preds)
    frame_ids = _cached_frames(scene_cache / camera_ids[0], wanted)
    stride = frame_ids[1] - frame_ids[0] if len(frame_ids) > 1 else 1
    fps = args.fps if args.fps else max(FPS / stride, 1.0)
    out_dir = Path(args.out)
    written: list[Path] = []

    if args.view in ("camera", "both"):
        for camera_id in camera_ids:
            camera = calibration.by_id(camera_id)
            destination = out_dir / f"{args.scene}_{camera_id}.mp4"
            writer, visible = None, 0
            try:
                for frame_id in frame_ids:
                    path = scene_cache / camera_id / f"{frame_id:06d}.jpg"
                    image = cv2.imread(str(path))
                    if image is None:
                        raise FileNotFoundError(f"missing cached frame {path}")
                    visible += draw_camera_view(
                        image, camera, predictions.get(frame_id, []), args.color_by
                    )
                    if writer is None:
                        writer = _open_writer(destination, (image.shape[1], image.shape[0]), fps)
                    writer.write(image)
            finally:
                if writer is not None:
                    writer.release()
            print(f"wrote {destination} ({len(frame_ids)} frames, {visible} boxes drawn)")
            written.append(destination)

    if args.view in ("bev", "both"):
        boxes = np.asarray(
            [box[:2] for frame in frame_ids for _, _, box in predictions.get(frame, [])]
        ).reshape(-1, 2)
        view = BirdsEyeView(calibration, boxes, args.bev_size)
        destination = out_dir / f"{args.scene}_bev.mp4"
        writer = _open_writer(destination, (view.width, view.height), fps)
        try:
            for frame_id in frame_ids:
                writer.write(view.render(predictions.get(frame_id, []), args.color_by, frame_id))
        finally:
            writer.release()
        print(f"wrote {destination} ({len(frame_ids)} frames, {view.width}x{view.height})")
        written.append(destination)

    return written


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    args = parser(prog).parse_args(argv)
    written = run(args)
    print(f"{len(written)} clip(s) under {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
