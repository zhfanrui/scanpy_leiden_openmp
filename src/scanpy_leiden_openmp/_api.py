from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._backend import LeidenResult, run_igraph_backend, run_openmp_backend, run_python_backend


def _is_adata_like(obj: Any) -> bool:
    return hasattr(obj, "obsp") and hasattr(obj, "uns") and hasattr(obj, "obs")


def _neighbors_connectivities_key(adata: Any, neighbors_key: str) -> str:
    block = adata.uns.get(neighbors_key, {}) if hasattr(adata, "uns") else {}
    if isinstance(block, dict):
        return block.get("connectivities_key", "connectivities")
    return "connectivities"


def _extract_graph(
    adata_or_graph: Any,
    *,
    neighbors_key: str,
    obsp: str | None,
) -> tuple[sp.csr_matrix, bool]:
    if _is_adata_like(adata_or_graph):
        adata = adata_or_graph
        key = obsp or _neighbors_connectivities_key(adata, neighbors_key)
        if key not in adata.obsp:
            raise KeyError(f"Graph key {key!r} not found in adata.obsp")
        graph = adata.obsp[key]
        return _to_csr(graph), True

    return _to_csr(adata_or_graph), False


def _to_csr(graph: Any) -> sp.csr_matrix:
    if sp.isspmatrix_csr(graph):
        return graph
    if sp.issparse(graph):
        return graph.tocsr(copy=False)
    raise TypeError("Expected an AnnData-like object or scipy sparse matrix")


def _set_obs_labels(adata: Any, key_added: str, labels: np.ndarray) -> None:
    try:
        import pandas as pd

        adata.obs[key_added] = pd.Categorical(labels.astype(str))
    except Exception:
        adata.obs[key_added] = [str(x) for x in labels]


def _set_uns_params(adata: Any, params: dict[str, Any]) -> None:
    leiden_block = adata.uns.setdefault("leiden", {})
    if isinstance(leiden_block, dict):
        leiden_block["params"] = params


def _resolve_seed(random_state: int | None) -> int:
    return 0 if random_state is None else int(random_state)


def _do_fallback_scanpy(
    fallback: Callable[..., Any] | None,
    adata: Any,
    kwargs: dict[str, Any],
) -> Any:
    if fallback is None:
        raise RuntimeError("No Scanpy fallback function is available")
    return fallback(adata, **kwargs)


def leiden(
    adata_or_graph: Any,
    *,
    resolution: float = 1.0,
    random_state: int | None = 0,
    key_added: str = "leiden",
    neighbors_key: str = "neighbors",
    obsp: str | None = None,
    copy: bool = False,
    n_iterations: int = -1,
    partition_type: Any | None = None,
    backend: str = "openmp",
    n_threads: int | None = None,
    refine_with_igraph: bool = False,
    strict: bool = False,
    return_quality: bool = False,
    _scanpy_fallback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run Leiden clustering on an AnnData-like object or a sparse graph.

    Parameters
    ----------
    adata_or_graph
        AnnData-like object (with `obsp/obs/uns`) or scipy sparse matrix.
    backend
        One of `"openmp"`, `"igraph"`, `"python"`, `"leidenalg"`.
    refine_with_igraph
        Only used with `backend="openmp"`. If OpenMP succeeds, run an
        igraph/leidenalg refine pass using OpenMP labels as initial membership.
    strict
        If `True`, backend failures raise immediately. If `False`, use fallback.
    return_quality
        If `True`, include quality/iterations in return value.

    Returns
    -------
    Any
        For matrix input: labels (or labels + metadata when `return_quality=True`).
        For AnnData input: writes to `obs[key_added]` and `uns["leiden"]`, returns
        `None` unless `copy=True`.
    """
    if partition_type is not None:
        if strict:
            raise NotImplementedError("partition_type is not supported by this backend")
        warnings.warn("partition_type unsupported, fallback to Scanpy if available", stacklevel=2)
        if _is_adata_like(adata_or_graph):
            fb_kwargs = dict(
                resolution=resolution,
                random_state=random_state,
                key_added=key_added,
                neighbors_key=neighbors_key,
                obsp=obsp,
                copy=copy,
                n_iterations=n_iterations,
                partition_type=partition_type,
                **kwargs,
            )
            return _do_fallback_scanpy(_scanpy_fallback, adata_or_graph, fb_kwargs)
        raise NotImplementedError("partition_type fallback requires AnnData input")

    if copy and _is_adata_like(adata_or_graph) and hasattr(adata_or_graph, "copy"):
        target = adata_or_graph.copy()
    elif copy:
        target = deepcopy(adata_or_graph)
    else:
        target = adata_or_graph
    graph, is_adata = _extract_graph(target, neighbors_key=neighbors_key, obsp=obsp)

    seed = _resolve_seed(random_state)
    directed = kwargs.pop("directed", False)
    weighted = kwargs.pop("weighted", True)

    result: LeidenResult
    if backend == "openmp":
        try:
            result = run_openmp_backend(
                graph,
                resolution=float(resolution),
                seed=seed,
                n_iterations=int(n_iterations),
                directed=bool(directed),
                weighted=bool(weighted),
                n_threads=n_threads,
            )
            if refine_with_igraph:
                try:
                    result = run_igraph_backend(
                        graph,
                        resolution=float(resolution),
                        seed=seed,
                        n_iterations=int(n_iterations),
                        directed=bool(directed),
                        weighted=bool(weighted),
                        initial_membership=result.labels,
                    )
                except Exception as refine_exc:
                    if strict:
                        raise
                    warnings.warn(
                        f"igraph refine failed, keep openmp result: {refine_exc}",
                        stacklevel=2,
                    )
        except Exception as exc:
            if strict:
                raise
            warnings.warn(
                f"OpenMP backend failed, trying igraph/leidenalg fallback: {exc}",
                stacklevel=2,
            )
            try:
                result = run_igraph_backend(
                    graph,
                    resolution=float(resolution),
                    seed=seed,
                    n_iterations=int(n_iterations),
                    directed=bool(directed),
                    weighted=bool(weighted),
                )
            except Exception as ig_exc:
                if is_adata and _scanpy_fallback is not None:
                    warnings.warn(
                        f"igraph fallback failed, fallback to scanpy.leiden: {ig_exc}",
                        stacklevel=2,
                    )
                    fb_kwargs = dict(
                        resolution=resolution,
                        random_state=random_state,
                        key_added=key_added,
                        neighbors_key=neighbors_key,
                        obsp=obsp,
                        copy=copy,
                        n_iterations=n_iterations,
                        partition_type=partition_type,
                        **kwargs,
                    )
                    return _do_fallback_scanpy(_scanpy_fallback, target, fb_kwargs)
                warnings.warn(
                    f"igraph fallback failed, fallback to python backend: {ig_exc}",
                    stacklevel=2,
                )
                result = run_python_backend(
                    graph,
                    resolution=float(resolution),
                    seed=seed,
                    n_iterations=int(n_iterations),
                    directed=bool(directed),
                    weighted=bool(weighted),
                )
    elif backend == "igraph":
        result = run_igraph_backend(
            graph,
            resolution=float(resolution),
            seed=seed,
            n_iterations=int(n_iterations),
            directed=bool(directed),
            weighted=bool(weighted),
        )
    elif backend == "python":
        result = run_python_backend(
            graph,
            resolution=float(resolution),
            seed=seed,
            n_iterations=int(n_iterations),
            directed=bool(directed),
            weighted=bool(weighted),
        )
    elif backend == "leidenalg":
        if not is_adata:
            raise ValueError("backend='leidenalg' requires AnnData input")
        fb_kwargs = dict(
            resolution=resolution,
            random_state=random_state,
            key_added=key_added,
            neighbors_key=neighbors_key,
            obsp=obsp,
            copy=copy,
            n_iterations=n_iterations,
            partition_type=partition_type,
            **kwargs,
        )
        return _do_fallback_scanpy(_scanpy_fallback, target, fb_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if is_adata:
        _set_obs_labels(target, key_added=key_added, labels=result.labels)
        _set_uns_params(
            target,
            {
                "resolution": float(resolution),
                "random_state": int(seed),
                "n_iterations": int(n_iterations),
                "backend": result.backend,
                "n_threads": n_threads,
                "refine_with_igraph": bool(refine_with_igraph),
            },
        )
        if copy:
            return target
        if return_quality:
            return result.quality, result.iterations_done
        return None

    if return_quality:
        return result.labels, result.quality, result.iterations_done
    return result.labels
