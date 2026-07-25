import os

import numpy as np

try:
    from turbovec import IdMapIndex
except Exception:
    IdMapIndex = None


def _as_2d_float32(vectors):
    """Coerce embedding chunks to a contiguous 2D float32 array (N, D).

    Accepts:
      - 2D arrays (N, D)
      - 1D single vector (D,) -> (1, D)
      - lists/sequences of 1D vectors
      - empty inputs -> (0, 0)
    Rejects / flattens higher-rank vectors by squeezing trailing singleton dims
    where safe; otherwise raises a clear error.
    """
    if vectors is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(vectors, (list, tuple)):
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        rows = []
        for item in vectors:
            row = np.asarray(item, dtype=np.float32)
            row = np.squeeze(row)
            if row.ndim == 0:
                raise ValueError("chunk vectors must be a 2D float32 array")
            if row.ndim > 1:
                row = row.reshape(-1)
            rows.append(row)
        lengths = {int(r.shape[0]) for r in rows}
        if len(lengths) != 1:
            raise ValueError("chunk vectors have inconsistent embedding dimensions")
        arr = np.ascontiguousarray(np.stack(rows, axis=0).astype(np.float32))
        return arr
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        if arr.ndim == 2:
            return np.ascontiguousarray(arr)
        return np.zeros((0, 0), dtype=np.float32)
    # Squeeze singleton dims such as (N, 1, D) or (1, D) carefully.
    while arr.ndim > 2 and 1 in arr.shape:
        # Prefer squeezing axis=1 when shape is (N, 1, D)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
            continue
        arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"chunk vectors must be a 2D float32 array (got shape {arr.shape})"
        )
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


class SegmentVectorIndex:
    def __init__(self, bit_width=4):
        self.bit_width = bit_width
        self.idx = None
        self.ids = np.zeros((0,), dtype=np.uint64)
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        self.path = None
        self.backend = "numpy"

    def build(self, vectors, ids, path=None):
        arr = _as_2d_float32(vectors)
        ext_ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
        if arr.shape[0] != ext_ids.shape[0]:
            raise ValueError("vectors and ids length mismatch")
        self.vecs = arr
        self.ids = ext_ids
        self.path = path
        self.idx = None
        self.backend = "numpy"
        if IdMapIndex is None or arr.size == 0 or arr.shape[0] == 0:
            return
        try:
            self.idx = IdMapIndex(dim=arr.shape[1], bit_width=self.bit_width)
            self.idx.add_with_ids(np.ascontiguousarray(arr), ext_ids)
            self.idx.prepare()
            self.backend = "turbovec"
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                self.idx.write(path)
        except Exception:
            self.idx = None
            self.backend = "numpy"

    def build_merged(self, chunks, path=None):
        vec_parts = []
        id_parts = []
        for vecs, ids in chunks or []:
            ext_ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
            if ext_ids.size == 0:
                continue
            arr = np.asarray(vecs) if not isinstance(vecs, (list, tuple)) else None
            if arr is not None and arr.size == 0:
                continue
            arr = _as_2d_float32(vecs)
            if arr.size == 0 or arr.shape[0] == 0:
                continue
            if arr.shape[0] != ext_ids.shape[0]:
                raise ValueError("chunk vectors and ids length mismatch")
            vec_parts.append(arr)
            id_parts.append(ext_ids)
        if not vec_parts:
            self.build(np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.uint64), path=path)
            return
        dims = {int(v.shape[1]) for v in vec_parts}
        if len(dims) != 1:
            raise ValueError("chunk vectors have inconsistent embedding dimensions")
        merged_vecs = np.ascontiguousarray(np.concatenate(vec_parts, axis=0))
        merged_ids = np.ascontiguousarray(np.concatenate(id_parts, axis=0))
        self.build(merged_vecs, merged_ids, path=path)

    def search(self, query, k):
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.backend == "turbovec" and self.idx is not None and len(self.ids):
            scores, ids = self.idx.search(np.ascontiguousarray(q), k=k)
            return np.asarray(scores[0], dtype=np.float32), np.asarray(ids[0], dtype=np.uint64)
        if not len(self.ids):
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.uint64)
        sims = self.vecs @ q[0]
        order = np.argsort(-sims)[:k]
        return sims[order].astype(np.float32), self.ids[order]
