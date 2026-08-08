"""Tests for the box geometry and the visualize subcommand."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from sparseqid.cli.visualize import (
    BirdsEyeView,
    color_for,
    parse_frame_range,
    read_predictions,
)
from sparseqid.cli.visualize import main as visualize_main
from sparseqid.geometry import BOTTOM_FACE, boxes_to_corners, euler_to_rotation


def test_axis_aligned_box_corners_span_its_extent() -> None:
    corners = boxes_to_corners(np.array([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]]))[0]

    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(corners.max(axis=0), [2.0, 4.0, 6.0])
    # Corners 0-3 form the bottom face, 4-7 the top.
    np.testing.assert_allclose(corners[list(BOTTOM_FACE), 2], 0.0)
    np.testing.assert_allclose(corners[4:, 2], 6.0)


def test_quarter_turn_yaw_swaps_the_footprint_extent() -> None:
    upright = boxes_to_corners(np.array([[0.0, 0.0, 0.0, 2.0, 6.0, 1.0, 0.0]]))[0]
    turned = boxes_to_corners(np.array([[0.0, 0.0, 0.0, 2.0, 6.0, 1.0, np.pi / 2]]))[0]

    np.testing.assert_allclose(np.ptp(upright[:, 0]), 2.0)
    np.testing.assert_allclose(np.ptp(upright[:, 1]), 6.0)
    np.testing.assert_allclose(np.ptp(turned[:, 0]), 6.0, atol=1e-9)
    np.testing.assert_allclose(np.ptp(turned[:, 1]), 2.0, atol=1e-9)


def test_yaw_rotation_matches_the_evaluator_convention() -> None:
    # Rz(90 deg) sends +x to +y.
    rotated = euler_to_rotation(0.0, 0.0, np.pi / 2) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_seven_and_nine_column_boxes_agree_when_pitch_and_roll_are_zero() -> None:
    seven = boxes_to_corners(np.array([[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 0.4]]))
    nine = boxes_to_corners(np.array([[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.4]]))
    np.testing.assert_allclose(seven, nine)


def test_rejects_box_arrays_that_are_not_seven_or_nine_columns() -> None:
    with pytest.raises(ValueError, match="7 or 9 box columns"):
        boxes_to_corners(np.zeros((1, 8)))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0:4", [0, 1, 2, 3]), ("0:10:3", [0, 3, 6, 9]), ("5:7", [5, 6])],
)
def test_frame_range_parsing(text, expected) -> None:
    assert list(parse_frame_range(text)) == expected


@pytest.mark.parametrize("text", ["0", "0:0", "5:1", "a:b", "0:10:0", "0:1:2:3"])
def test_frame_range_rejects_malformed_input(text) -> None:
    with pytest.raises(ValueError):
        parse_frame_range(text)


def test_predictions_are_grouped_by_frame(tmp_path) -> None:
    path = tmp_path / "preds.txt"
    path.write_text(
        "20 0 7 0 1.0 2.0 0.5 0.6 0.7 1.8 0.1\n"
        "20 1 8 0 3.0 4.0 0.5 0.6 0.7 1.8 0.2\n"
        "\n"
        "20 0 7 3 1.5 2.5 0.5 0.6 0.7 1.8 0.3\n"
    )
    per_frame = read_predictions(path)

    assert sorted(per_frame) == [0, 3]
    assert len(per_frame[0]) == 2
    class_id, object_id, box = per_frame[3][0]
    assert (class_id, object_id) == (0, 7)
    np.testing.assert_allclose(box[:2], [1.5, 2.5])


def test_predictions_reject_a_file_that_is_not_the_submission_format(tmp_path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("20 0 7 0 1.0\n")
    with pytest.raises(ValueError, match="expected 11 columns"):
        read_predictions(path)


def test_colour_is_stable_and_distinguishes_tracks() -> None:
    assert color_for(0, 5, "id") == color_for(0, 5, "id")
    assert color_for(0, 5, "id") != color_for(0, 6, "id")
    # Colouring by class ignores the track id.
    assert color_for(3, 5, "class") == color_for(3, 99, "class")


def _calibration(path, camera_ids, width=64, height=48) -> None:
    """Write a minimal calibration.json with cameras looking down the +z axis."""
    sensors = []
    for index, camera_id in enumerate(camera_ids):
        intrinsic = [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]]
        # Camera sits at z = -20 looking at the origin, offset along x per camera.
        extrinsic = [[1, 0, 0, -float(index)], [0, 1, 0, 0], [0, 0, 1, 20.0]]
        sensors.append(
            {
                "type": "camera",
                "id": camera_id,
                "intrinsicMatrix": intrinsic,
                "extrinsicMatrix": extrinsic,
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "attributes": [
                    {"name": "frameWidth", "value": str(width)},
                    {"name": "frameHeight", "value": str(height)},
                    {"name": "fps", "value": "30"},
                ],
            }
        )
    path.write_text(json.dumps({"sensors": sensors}))


def _frame_cache(root, frame_ids, width=64, height=48) -> None:
    root.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    for frame_id in frame_ids:
        cv2.imwrite(str(root / f"{frame_id:06d}.jpg"), image)


def test_bev_flips_y_so_north_is_up() -> None:
    class _Calibration:
        cameras = [type("C", (), {"center": np.array([0.0, 0.0, 3.0])})()]

    boxes = np.array([[-10.0, -10.0], [10.0, 10.0]])
    view = BirdsEyeView(_Calibration(), boxes, size=200)
    pixels = view.to_pixels(np.array([[0.0, -5.0], [0.0, 5.0]]))

    # A larger world y must map to a smaller row index.
    assert pixels[1][1] < pixels[0][1]
    # x is not flipped.
    assert (
        view.to_pixels(np.array([[5.0, 0.0]]))[0][0] > view.to_pixels(np.array([[-5.0, 0.0]]))[0][0]
    )


def test_visualize_writes_playable_clips_for_both_views(tmp_path, capsys) -> None:
    data_root, cache_root = tmp_path / "data", tmp_path / "cache"
    scene_dir = data_root / "val" / "Warehouse_020"
    scene_dir.mkdir(parents=True)
    _calibration(scene_dir / "calibration.json", ["Camera_0000"])

    # An --every 3 cache: frame ids are 0, 3, 6, 9 with gaps between them.
    frame_ids = [0, 3, 6, 9]
    _frame_cache(cache_root / "val" / "Warehouse_020" / "Camera_0000", frame_ids)

    predictions = tmp_path / "preds.txt"
    predictions.write_text(
        "".join(
            f"20 0 1 {frame} 0.0 {0.1 * frame:.3f} 0.0 0.8 0.8 1.7 0.0\n" for frame in frame_ids
        )
    )

    out = tmp_path / "viz"
    visualize_main(
        [
            "--preds",
            str(predictions),
            "--scene",
            "Warehouse_020",
            "--split",
            "val",
            "--data-root",
            str(data_root),
            "--cache-root",
            str(cache_root),
            "--frames",
            "0:12",
            "--out",
            str(out),
            "--bev-size",
            "160",
        ]
    )

    camera_clip = out / "Warehouse_020_Camera_0000.mp4"
    bev_clip = out / "Warehouse_020_bev.mp4"
    for clip in (camera_clip, bev_clip):
        assert clip.is_file(), f"{clip} was not written"
        assert clip.stat().st_size > 0, f"{clip} is empty"
        capture = cv2.VideoCapture(str(clip))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frame_ids)
            # A 3-frame stride means real-time playback is 10 fps, not 30.
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.5)
        finally:
            capture.release()
    assert "boxes drawn" in capsys.readouterr().out


def test_visualize_reports_an_empty_frame_range(tmp_path) -> None:
    data_root, cache_root = tmp_path / "data", tmp_path / "cache"
    scene_dir = data_root / "val" / "Warehouse_020"
    scene_dir.mkdir(parents=True)
    _calibration(scene_dir / "calibration.json", ["Camera_0000"])
    _frame_cache(cache_root / "val" / "Warehouse_020" / "Camera_0000", [0, 1, 2])
    predictions = tmp_path / "preds.txt"
    predictions.write_text("20 0 1 0 0.0 0.0 0.0 0.8 0.8 1.7 0.0\n")

    with pytest.raises(ValueError, match="no cached frames in range"):
        visualize_main(
            [
                "--preds",
                str(predictions),
                "--scene",
                "Warehouse_020",
                "--data-root",
                str(data_root),
                "--cache-root",
                str(cache_root),
                "--frames",
                "500:600",
                "--out",
                str(tmp_path / "viz"),
            ]
        )
