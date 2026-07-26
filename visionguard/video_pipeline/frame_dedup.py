"""Lossless consecutive-frame deduplication for video indexing."""
from __future__ import annotations

import numpy as np


def is_exact_duplicate(previous: np.ndarray | None, current: np.ndarray) -> bool:
    """Return true only when two decoded frames have identical pixels.

    This intentionally does not use perceptual hashes, motion thresholds, or
    time gaps: any changed pixel remains indexable evidence.
    """
    return bool(
        previous is not None
        and previous.shape == current.shape
        and previous.dtype == current.dtype
        and np.array_equal(previous, current)
    )
