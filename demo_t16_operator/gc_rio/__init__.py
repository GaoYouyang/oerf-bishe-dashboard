"""Small, leakage-resistant development data for v5h GC-RIO."""

from .data import (
    GCRIODatasetBundle,
    build_dataset,
    paired_difference_mad,
)

__all__ = ["GCRIODatasetBundle", "build_dataset", "paired_difference_mad"]
