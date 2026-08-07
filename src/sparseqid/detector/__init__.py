"""Recurrent TAO Sparse4D detector integration used by SparseQID."""

from .build import DEFAULT_CONFIG_DIR, build_architecture, build_paper_detector
from .sparse4d import Sparse4D

__all__ = ["DEFAULT_CONFIG_DIR", "Sparse4D", "build_architecture", "build_paper_detector"]
