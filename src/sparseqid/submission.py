# SPDX-License-Identifier: Apache-2.0

"""Write the AI City 11-column submission format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    """One world-frame box observation belonging to a track at a frame."""

    scene_id: int
    class_id: int
    object_id: int
    frame_id: int
    x: float
    y: float
    z: float
    w: float
    l: float
    h: float
    yaw: float

    def to_line(self) -> str:
        return (
            f"{self.scene_id} {self.class_id} {self.object_id} {self.frame_id} "
            f"{self.x:.6f} {self.y:.6f} {self.z:.6f} "
            f"{self.w:.6f} {self.l:.6f} {self.h:.6f} {self.yaw:.6f}"
        )


def write_tracks(tracks: list[Track], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(f"{track.to_line()}\n" for track in tracks))
