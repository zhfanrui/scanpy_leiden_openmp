from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scanpy_leiden_openmp import patch_scanpy, unpatch_scanpy


@dataclass
class RunResult:
    labels: np.ndarray
    elapsed_s: float


def _to_int_codes(series) -> np.ndarray:
    # Make sure categorical/string labels are converted to comparable int ids.
    return np.asarray(series.astype("category").cat.codes, dtype=np.int64)


def run_scanpy_leiden(
    adata,
    *,
    resolution: float,
    random_state: int,
    n_iterations: int,
    key_added: str,
    neighbors_key: str,
    obsp: str | None,
) -> RunResult:
    """Run baseline Scanpy Leiden and return labels + elapsed time."""
    a = adata.copy()
    t0 = time.perf_counter()
    sc.tl.leiden(
        a,
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
        key_added=key_added,
        neighbors_key=neighbors_key,
        obsp=obsp,
    )
    elapsed = time.perf_counter() - t0
    return RunResult(labels=_to_int_codes(a.obs[key_added]), elapsed_s=elapsed)


def run_patch_openmp_leiden(
    adata,
    *,
    resolution: float,
    random_state: int,
    n_iterations: int,
    key_added: str,
    neighbors_key: str,
    obsp: str | None,
    n_threads: int | None,
    refine_with_igraph: bool,
    strict: bool,
) -> RunResult:
    """Run patch-mode OpenMP Leiden and return labels + elapsed time."""
    a = adata.copy()
    t0 = time.perf_counter()
    patch_scanpy()
    try:
        sc.tl.leiden(
            a,
            resolution=resolution,
            random_state=random_state,
            n_iterations=n_iterations,
            key_added=key_added,
            neighbors_key=neighbors_key,
            obsp=obsp,
            backend="openmp",
            n_threads=n_threads,
            refine_with_igraph=refine_with_igraph,
            strict=strict,
        )
    finally:
        unpatch_scanpy()
    elapsed = time.perf_counter() - t0
    return RunResult(labels=_to_int_codes(a.obs[key_added]), elapsed_s=elapsed)


def ensure_neighbors(adata, *, neighbors_key: str, n_neighbors: int, n_pcs: int) -> None:
    """Ensure neighbors graph exists; create it when missing."""
    if neighbors_key in adata.uns and isinstance(adata.uns[neighbors_key], dict):
        return
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, key_added=neighbors_key)


def ensure_umap(adata, *, neighbors_key: str) -> None:
    """Ensure UMAP embedding exists; compute it when missing."""
    if "X_umap" in adata.obsm:
        return
    sc.tl.umap(adata, neighbors_key=neighbors_key)


def save_visualization(adata, *, baseline_key: str, patch_key: str, output_png: str) -> None:
    """Save side-by-side UMAP panels for baseline and patch results."""
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    sc.pl.umap(
        adata,
        color=[baseline_key, patch_key],
        wspace=0.35,
        show=False,
        save=False,
    )
    import matplotlib.pyplot as plt

    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Scanpy leidenalg vs OpenMP backend on the same AnnData.")
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--neighbors-key", default="neighbors")
    parser.add_argument("--obsp", default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument(
        "--refine-with-igraph",
        action="store_true",
        help="After OpenMP result, run one igraph refine pass with initial membership.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if OpenMP backend is unavailable")
    parser.add_argument(
        "--out-plot",
        default="compare_leiden_umap.png",
        help="Output path for UMAP comparison plot",
    )
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    ensure_neighbors(
        adata,
        neighbors_key=args.neighbors_key,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
    )

    baseline = run_scanpy_leiden(
        adata,
        resolution=args.resolution,
        random_state=args.random_state,
        n_iterations=args.n_iterations,
        key_added="_baseline_leiden",
        neighbors_key=args.neighbors_key,
        obsp=args.obsp,
    )

    patched_openmp = run_patch_openmp_leiden(
        adata,
        resolution=args.resolution,
        random_state=args.random_state,
        n_iterations=args.n_iterations,
        key_added="_openmp_leiden",
        neighbors_key=args.neighbors_key,
        obsp=args.obsp,
        n_threads=args.n_threads,
        refine_with_igraph=args.refine_with_igraph,
        strict=args.strict,
    )

    ari = adjusted_rand_score(baseline.labels, patched_openmp.labels)
    nmi = normalized_mutual_info_score(baseline.labels, patched_openmp.labels)
    speedup = baseline.elapsed_s / patched_openmp.elapsed_s if patched_openmp.elapsed_s > 0 else float("inf")

    adata.obs["_baseline_leiden"] = baseline.labels.astype(str)
    adata.obs["_patch_openmp_leiden"] = patched_openmp.labels.astype(str)
    ensure_umap(adata, neighbors_key=args.neighbors_key)
    save_visualization(
        adata,
        baseline_key="_baseline_leiden",
        patch_key="_patch_openmp_leiden",
        output_png=args.out_plot,
    )

    print("=== Leiden Comparison ===")
    print(f"cells: {adata.n_obs}")
    print(f"baseline clusters (scanpy): {len(np.unique(baseline.labels))}")
    print(f"patch(openmp) clusters: {len(np.unique(patched_openmp.labels))}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")
    print(f"baseline time: {baseline.elapsed_s:.6f} s")
    print(f"patch(openmp) time: {patched_openmp.elapsed_s:.6f} s")
    print(f"speedup (baseline/patch-openmp): {speedup:.3f}x")
    print(f"saved plot: {Path(args.out_plot).resolve()}")


if __name__ == "__main__":
    main()
