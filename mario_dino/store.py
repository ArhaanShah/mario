"""Append-only, resumable array store: a raw memmap file + index.csv + done mask.

Layout of ``root``:

* ``features.bin``  raw array, shape ``[n, *row_shape]`` in ``dtype`` (fp16 by default), row-major
* ``index.csv``     one ``image`` name per row, in row order
* ``done.npy``      bool ``[n]``, rewritten atomically after each finished chunk
* ``meta.json``     row shape, dtype and the identity (encoder, weights hash, preprocessing, ...)

New images are appended by growing the file, so one store can serve several CSVs and tasks without
re-encoding shared images. Opening a store with a different identity raises instead of mixing features.
Used for pooled features (row shape ``[2, 2D]``) and for partial fine-tune token caches.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

DATA_FILE = "features.bin"
INDEX_FILE = "index.csv"
DONE_FILE = "done.npy"
META_FILE = "meta.json"


class StoreMismatch(RuntimeError):
    pass


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class FeatureStore:
    def __init__(self, root: str, row_shape: Sequence[int], dtype: str = "float16", identity: dict | None = None):
        """Open ``root``, creating it if needed; an existing store must match row shape, dtype and identity."""
        self.root = root
        self.row_shape = tuple(int(s) for s in row_shape)
        self.dtype = np.dtype(dtype)
        self.identity = json.loads(json.dumps(identity or {}, sort_keys=True))
        os.makedirs(root, exist_ok=True)
        meta_path = os.path.join(root, META_FILE)
        meta = {"row_shape": list(self.row_shape), "dtype": self.dtype.name, "identity": self.identity}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                old = json.load(f)
            if old != meta:
                raise StoreMismatch(
                    f"store {root!r} was built with {old}, requested {meta}; use another store_dir"
                )
        else:
            _write_json(meta_path, meta)
        idx_path = os.path.join(root, INDEX_FILE)
        self.images: list[str] = (
            pd.read_csv(idx_path, dtype={"image": str})["image"].tolist() if os.path.isfile(idx_path) else []
        )
        self._row = {im: i for i, im in enumerate(self.images)}
        done_path = os.path.join(root, DONE_FILE)
        done = np.load(done_path) if os.path.isfile(done_path) else np.zeros(0, dtype=bool)
        # a crash between growing the index and the done mask leaves the mask short: pad with False
        self.done = np.zeros(len(self.images), dtype=bool)
        self.done[: min(len(done), len(self.images))] = done[: len(self.images)]
        self._mm: np.memmap | None = None
        self._resize_file(len(self.images))

    # ---- layout -------------------------------------------------------------------------------
    @property
    def row_bytes(self) -> int:
        return int(np.prod(self.row_shape)) * self.dtype.itemsize

    def __len__(self) -> int:
        return len(self.images)

    def _close(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            self._mm = None
            gc.collect()  # Windows cannot resize a file while a mapping is alive

    def _resize_file(self, n: int) -> None:
        path = os.path.join(self.root, DATA_FILE)
        want = n * self.row_bytes
        have = os.path.getsize(path) if os.path.isfile(path) else -1
        if have != want:
            self._close()
            with open(path, "ab" if have >= 0 else "wb") as f:
                f.truncate(want)

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            if len(self.images) == 0:
                raise ValueError("store is empty")
            self._mm = np.memmap(
                os.path.join(self.root, DATA_FILE),
                dtype=self.dtype,
                mode="r+",
                shape=(len(self.images), *self.row_shape),
            )
        return self._mm

    # ---- index --------------------------------------------------------------------------------
    def ensure(self, images: Sequence[str]) -> np.ndarray:
        """Rows for ``images``, appending any that are new."""
        new = [im for im in dict.fromkeys(str(i) for i in images) if im not in self._row]
        if new:
            self._close()
            n0 = len(self.images)
            self._resize_file(n0 + len(new))
            self.images.extend(new)
            self._row.update({im: n0 + i for i, im in enumerate(new)})
            tmp = os.path.join(self.root, INDEX_FILE + ".tmp")
            pd.DataFrame({"image": self.images}).to_csv(tmp, index=False, lineterminator="\n")
            os.replace(tmp, os.path.join(self.root, INDEX_FILE))
            self.done = np.concatenate([self.done, np.zeros(len(new), dtype=bool)])
            self._save_done()
        return self.rows(images)

    def rows(self, images: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self._row[str(im)] for im in images], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"image {e.args[0]!r} is not in store {self.root!r}") from None

    def contains(self, images: Sequence[str]) -> np.ndarray:
        return np.array([str(im) in self._row for im in images], dtype=bool)

    # ---- data ---------------------------------------------------------------------------------
    def pending(self, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(len(self.images)) if rows is None else np.asarray(rows)
        return rows[~self.done[rows]]

    def write(self, rows: np.ndarray, values: np.ndarray) -> None:
        mm = self._memmap()
        mm[np.asarray(rows)] = np.asarray(values, dtype=self.dtype).reshape(len(rows), *self.row_shape)
        mm.flush()

    def mark_done(self, rows: np.ndarray) -> None:
        self.done[np.asarray(rows)] = True
        self._save_done()

    def _save_done(self) -> None:
        tmp = os.path.join(self.root, DONE_FILE + ".tmp.npy")
        np.save(tmp, self.done)
        os.replace(tmp, os.path.join(self.root, DONE_FILE))

    def get_rows(self, rows: np.ndarray, require_done: bool = True) -> np.ndarray:
        rows = np.asarray(rows)
        if require_done and not self.done[rows].all():
            missing = [self.images[r] for r in rows[~self.done[rows]][:5]]
            raise RuntimeError(f"rows not computed yet in {self.root!r}, e.g. {missing}; run extraction")
        return np.array(self._memmap()[rows])

    def get(self, images: Sequence[str], require_done: bool = True) -> np.ndarray:
        return self.get_rows(self.rows(images), require_done)

    def load_all(self) -> np.ndarray:
        """Every row as an in-memory array (rows not yet computed are zeros)."""
        if not self.images:
            return np.zeros((0, *self.row_shape), dtype=self.dtype)
        return np.array(self._memmap())

    def close(self) -> None:
        self._close()
