from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp


try:
    from . import _core  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _core = None


@dataclass
class LeidenResult:
    labels: np.ndarray
    quality: float
    iterations_done: int
    backend: str


def openmp_available() -> bool:
    return _core is not None


def run_openmp_backend(
    graph: sp.csr_matrix,
    *,
    resolution: float,
    seed: int,
    n_iterations: int,
    directed: bool,
    weighted: bool,
    n_threads: Optional[int],
) -> LeidenResult:
    """Run the compiled OpenMP backend."""
    if _core is None:
        raise RuntimeError("OpenMP backend extension is unavailable")

    labels, quality, iterations_done = _core.run_leiden_csr(
        graph.indptr.astype(np.int64, copy=False),
        graph.indices.astype(np.int64, copy=False),
        graph.data.astype(np.float64, copy=False),
        graph.shape[0],
        float(resolution),
        int(seed),
        int(n_iterations),
        bool(directed),
        bool(weighted),
        int(n_threads if n_threads is not None else -1),
    )
    return LeidenResult(
        labels=np.asarray(labels, dtype=np.int64),
        quality=float(quality),
        iterations_done=int(iterations_done),
        backend="openmp",
    )


def run_python_backend(
    graph: sp.csr_matrix,
    *,
    resolution: float = 1.0,
    seed: int = 0,
    n_iterations: int = -1,
    directed: bool = False,
    weighted: bool = True,
) -> LeidenResult:
    """Python-level fallback path (currently delegated to igraph/leidenalg)."""
    # Fallback path is igraph/leidenalg.
    return run_igraph_backend(
        graph,
        resolution=resolution,
        seed=seed,
        n_iterations=n_iterations,
        directed=directed,
        weighted=weighted,
    )


def run_igraph_backend(
    graph: sp.csr_matrix,
    *,
    resolution: float,
    seed: int,
    n_iterations: int,
    directed: bool,
    weighted: bool,
    initial_membership: Optional[np.ndarray] = None,
) -> LeidenResult:
    """
    Run igraph/leidenalg backend.

    Parameters
    ----------
    initial_membership
        Optional initial community labels for refinement.
    """
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("igraph/leidenalg backend is unavailable") from exc

    coo = graph.tocoo(copy=False)
    if directed:
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        vals = np.asarray(coo.data, dtype=np.float64)
    else:
        mask = coo.row <= coo.col
        rows = np.asarray(coo.row[mask], dtype=np.int64)
        cols = np.asarray(coo.col[mask], dtype=np.int64)
        vals = np.asarray(coo.data[mask], dtype=np.float64)

    g = ig.Graph(n=graph.shape[0], edges=list(zip(rows.tolist(), cols.tolist())), directed=directed)
    weights = vals.tolist() if weighted else None
    membership = None
    if initial_membership is not None:
        membership = np.asarray(initial_membership, dtype=np.int64).tolist()
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
        initial_membership=membership,
    )
    quality = float(partition.quality())
    labels = np.asarray(partition.membership, dtype=np.int64)
    return LeidenResult(
        labels=labels,
        quality=quality,
        iterations_done=int(n_iterations if n_iterations > 0 else -1),
        backend="igraph",
    )
