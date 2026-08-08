# SPDX-License-Identifier: Apache-2.0

"""World-frame 3D box geometry.

The submission format stores boxes as ``[x, y, z, w, l, h, yaw]`` in world
coordinates. Turning those into corners is needed to project a box into a
camera or to draw its footprint from above.
"""

from __future__ import annotations

import numpy as np

# Unit cube corners: bottom face 0-3, top face 4-7 directly above 0-3.
_UNIT_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)

#: The twelve edges of a box, as index pairs into the corner array.
BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

#: Corner indices of the ground-plane face, in winding order.
BOTTOM_FACE: tuple[int, int, int, int] = (0, 1, 2, 3)


def euler_to_rotation(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(roll) @ Rx(pitch)``, the convention the evaluator uses."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(roll), np.sin(roll)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes to their eight world-frame corners.

    Args:
        boxes: ``(B, 7)`` as ``[x, y, z, w, l, h, yaw]``, or ``(B, 9)`` as
            ``[x, y, z, w, l, h, pitch, roll, yaw]``. A single box is accepted.

    Returns:
        ``(B, 8, 3)`` corners, ordered to match :data:`BOX_EDGES`.
    """
    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    if boxes.shape[1] == 7:
        padding = np.zeros((len(boxes), 2))
        boxes = np.concatenate([boxes[:, :6], padding, boxes[:, 6:7]], axis=1)
    elif boxes.shape[1] != 9:
        raise ValueError(f"expected 7 or 9 box columns, got {boxes.shape[1]}")

    centers, sizes = boxes[:, :3], boxes[:, 3:6]
    corners = _UNIT_CORNERS[None] * sizes[:, None, :] - sizes[:, None, :] / 2.0
    rotations = np.stack(
        [euler_to_rotation(pitch, roll, yaw) for pitch, roll, yaw in boxes[:, 6:9]]
    )
    # Rotate each corner by its box rotation, then translate to the box centre.
    return np.einsum("bij,bkj->bki", rotations, corners) + centers[:, None, :]
