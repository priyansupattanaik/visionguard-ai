import numpy as np

from visionguard.video_pipeline.frame_dedup import is_exact_duplicate


def test_only_identical_consecutive_pixels_are_duplicates():
    first = np.zeros((3, 4, 3), dtype=np.uint8)
    same = first.copy()
    changed = first.copy()
    changed[1, 2, 0] = 1

    assert is_exact_duplicate(first, same) is True
    assert is_exact_duplicate(first, changed) is False
    assert is_exact_duplicate(None, first) is False
