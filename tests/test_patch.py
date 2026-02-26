from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scanpy_leiden_openmp import patch_scanpy, unpatch_scanpy


class FakeAnnData:
    def __init__(self) -> None:
        rows = np.array([0, 1], dtype=np.int64)
        cols = np.array([1, 0], dtype=np.int64)
        data = np.ones_like(rows, dtype=np.float64)
        self.obsp = {"connectivities": sp.csr_matrix((data, (rows, cols)), shape=(2, 2))}
        self.uns = {"neighbors": {"connectivities_key": "connectivities"}}
        self.obs = pd.DataFrame(index=np.arange(2))


def test_patch_and_unpatch_scanpy() -> None:
    fake_scanpy = types.SimpleNamespace()
    fake_scanpy.tl = types.SimpleNamespace()

    def original(adata, **kwargs):
        return "original-called"

    fake_scanpy.tl.leiden = original
    sys.modules["scanpy"] = fake_scanpy

    patch_scanpy()
    adata = FakeAnnData()
    fake_scanpy.tl.leiden(adata, backend="python")
    assert "leiden" in adata.obs.columns

    fallback_value = fake_scanpy.tl.leiden(adata, backend="leidenalg")
    assert fallback_value == "original-called"

    unpatch_scanpy()
    assert fake_scanpy.tl.leiden is original

    del sys.modules["scanpy"]
