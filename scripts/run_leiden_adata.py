from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

from scanpy_leiden_openmp import leiden


def ensure_neighbors(adata, *, neighbors_key: str, n_neighbors: int, n_pcs: int) -> None:
    """Ensure neighbors graph exists for Leiden."""
    if neighbors_key in adata.uns and isinstance(adata.uns[neighbors_key], dict):
        return
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, key_added=neighbors_key)


def default_output_h5ad(input_h5ad: Path) -> Path:
    return input_h5ad.with_name(f"{input_h5ad.stem}.leiden.h5ad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read AnnData, run Leiden, and write outputs.")
    parser.add_argument("--h5ad", required=True, help="Input .h5ad path")
    parser.add_argument(
        "--output-h5ad",
        default=None,
        help="Output .h5ad path (default: <input>.leiden.h5ad)",
    )
    parser.add_argument("--output-csv", default=None, help="Optional cluster CSV output path")
    parser.add_argument("--key-added", default="leiden", help="obs column name to store clusters")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--neighbors-key", default="neighbors")
    parser.add_argument(
        "--obsp",
        default=None,
        help="Graph key in adata.obsp; if set, skip auto neighbors",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument(
        "--backend",
        choices=["openmp", "igraph", "python", "leidenalg"],
        default="openmp",
    )
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--refine-with-igraph", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    input_h5ad = Path(args.h5ad)
    output_h5ad = Path(args.output_h5ad) if args.output_h5ad else default_output_h5ad(input_h5ad)

    adata = sc.read_h5ad(input_h5ad)
    if args.obsp is None:
        ensure_neighbors(
            adata,
            neighbors_key=args.neighbors_key,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
        )

    leiden(
        adata,
        resolution=args.resolution,
        random_state=args.random_state,
        n_iterations=args.n_iterations,
        key_added=args.key_added,
        neighbors_key=args.neighbors_key,
        obsp=args.obsp,
        backend=args.backend,
        n_threads=args.n_threads,
        refine_with_igraph=args.refine_with_igraph,
        strict=args.strict,
    )

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        adata.obs[[args.key_added]].to_csv(output_csv)
        print(f"saved csv: {output_csv.resolve()}")

    print(f"cells: {adata.n_obs}")
    print(f"cluster_key: {args.key_added}")
    print(f"saved h5ad: {output_h5ad.resolve()}")


if __name__ == "__main__":
    main()
