from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scanpy_leiden_openmp import leiden
from scanpy_leiden_openmp._backend import LeidenResult


class FakeAnnData:
    def __init__(self, graph: sp.csr_matrix) -> None:
        self.obsp = {"connectivities": graph}
        self.uns = {"neighbors": {"connectivities_key": "connectivities"}}
        self.obs = pd.DataFrame(index=np.arange(graph.shape[0]))

    def copy(self) -> "FakeAnnData":
        copied = FakeAnnData(self.obsp["connectivities"].copy())
        copied.uns = {k: (v.copy() if isinstance(v, dict) else v) for k, v in self.uns.items()}
        copied.obs = self.obs.copy(deep=True)
        return copied


def _two_component_graph() -> sp.csr_matrix:
    # component A: 0-1, component B: 2-3
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 0, 3, 2])
    data = np.ones_like(rows, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 4))


def test_matrix_input_returns_labels() -> None:
    graph = _two_component_graph()
    labels = leiden(graph, backend="python")
    assert labels.shape == (4,)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_adata_like_writes_obs_and_uns() -> None:
    graph = _two_component_graph()
    adata = FakeAnnData(graph)
    result = leiden(adata, key_added="clusters", backend="python")
    assert result is None
    assert "clusters" in adata.obs.columns
    assert "leiden" in adata.uns
    assert adata.uns["leiden"]["params"]["backend"] == "igraph"


def test_copy_returns_new_object() -> None:
    graph = _two_component_graph()
    adata = FakeAnnData(graph)
    copied = leiden(adata, copy=True, key_added="clusters", backend="python")
    assert isinstance(copied, FakeAnnData)
    assert "clusters" in copied.obs.columns
    assert "clusters" not in adata.obs.columns


def test_openmp_fallback_when_unavailable() -> None:
    graph = _two_component_graph()
    labels = leiden(graph, backend="openmp", strict=False)
    assert labels.shape == (4,)


def test_openmp_refine_with_igraph_uses_initial_membership(monkeypatch) -> None:
    import scanpy_leiden_openmp._api as api

    graph = _two_component_graph()
    openmp_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    refined_labels = np.array([1, 1, 0, 0], dtype=np.int64)
    captured: dict[str, np.ndarray | None] = {"initial": None}

    def fake_openmp_backend(*args, **kwargs):
        _ = (args, kwargs)
        return LeidenResult(labels=openmp_labels, quality=1.0, iterations_done=1, backend="openmp")

    def fake_igraph_backend(*args, **kwargs):
        _ = args
        captured["initial"] = kwargs.get("initial_membership")
        return LeidenResult(labels=refined_labels, quality=2.0, iterations_done=2, backend="igraph")

    monkeypatch.setattr(api, "run_openmp_backend", fake_openmp_backend)
    monkeypatch.setattr(api, "run_igraph_backend", fake_igraph_backend)

    labels = leiden(graph, backend="openmp", strict=True, refine_with_igraph=True)
    assert np.array_equal(labels, refined_labels)
    assert captured["initial"] is not None
    assert np.array_equal(np.asarray(captured["initial"], dtype=np.int64), openmp_labels)
