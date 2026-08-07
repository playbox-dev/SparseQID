"""SparseQID: online 3D tracking by ID prediction over recurrent sparse queries."""

from __future__ import annotations

from typing import Any

__all__ = ["SparseQIDTracker", "track_clip"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .tracking import SparseQIDTracker, track_clip

        return {"SparseQIDTracker": SparseQIDTracker, "track_clip": track_clip}[name]
    raise AttributeError(name)
