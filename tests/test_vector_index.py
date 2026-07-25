"""Unit tests for vector index shape/dtype coercion."""
import numpy as np

from visionguard.video_pipeline.vector_index import SegmentVectorIndex, _as_2d_float32


def test_as_2d_from_1d_vector():
    arr = _as_2d_float32(np.random.randn(8).astype(np.float32))
    assert arr.ndim == 2
    assert arr.shape == (1, 8)
    assert arr.dtype == np.float32


def test_as_2d_from_list_of_1d():
    rows = [np.random.randn(8).astype(np.float32) for _ in range(3)]
    arr = _as_2d_float32(rows)
    assert arr.shape == (3, 8)


def test_as_2d_squeezes_3d_singleton():
    bad = np.random.randn(3, 1, 8).astype(np.float32)
    arr = _as_2d_float32(bad)
    assert arr.shape == (3, 8)


def test_build_merged_accepts_1d_chunk():
    idx = SegmentVectorIndex()
    vec = np.random.randn(8).astype(np.float32)
    ids = np.array([0], dtype=np.uint64)
    idx.build_merged([(vec, ids)])
    assert len(idx.ids) == 1
    assert idx.vecs.shape == (1, 8)


def test_build_merged_accepts_3d_chunk():
    idx = SegmentVectorIndex()
    vecs = np.random.randn(2, 1, 8).astype(np.float32)
    ids = np.arange(2, dtype=np.uint64)
    idx.build_merged([(vecs, ids)])
    assert idx.vecs.shape == (2, 8)


def test_build_merged_empty_chunks():
    idx = SegmentVectorIndex()
    idx.build_merged([])
    assert idx.vecs.size == 0
    scores, ids = idx.search(np.random.randn(8).astype(np.float32), k=3)
    assert len(scores) == 0
    assert len(ids) == 0


def test_build_merged_skips_empty_parts():
    idx = SegmentVectorIndex()
    good = np.random.randn(2, 8).astype(np.float32)
    idx.build_merged([
        (np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.uint64)),
        (good, np.arange(2, dtype=np.uint64)),
    ])
    assert idx.vecs.shape == (2, 8)
