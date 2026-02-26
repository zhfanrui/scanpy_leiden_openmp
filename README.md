# scanpy-leiden-openmp

A standalone Leiden backend package for Scanpy with a C++/OpenMP extension and Python patching hooks.

## Features

- `scanpy_leiden_openmp.leiden(...)`: run clustering on AnnData-like objects or CSR graph directly.
- `scanpy_leiden_openmp.patch_scanpy()`: patch `scanpy.tl.leiden` to use this backend.
- `scanpy_leiden_openmp.unpatch_scanpy()`: restore the original Scanpy function.
- Optional `refine_with_igraph=True`: after OpenMP clustering, run an igraph refine pass.
- `strict=True` to fail fast on unsupported options, `strict=False` to fallback.

## Quick start (uv)

```bash
uv sync --group dev
uv run pytest
uv build
```

### macOS + micromamba OpenMP setup

If CMake reports `Could NOT find OpenMP` on macOS, install OpenMP in your micromamba env and export paths before building:

```bash
micromamba activate <your-env>
micromamba install -c conda-forge llvm-openmp

export OpenMP_ROOT="$CONDA_PREFIX"
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"

uv build -v
```

## Usage

```python
import scanpy as sc
from scanpy_leiden_openmp import patch_scanpy, unpatch_scanpy

patch_scanpy()
sc.tl.leiden(
    adata,
    backend="openmp",
    n_threads=8,
    refine_with_igraph=True,
    strict=False,
)
unpatch_scanpy()
```

Or call the API directly:

```python
from scanpy_leiden_openmp import leiden

labels = leiden(
    csr_graph,
    backend="openmp",
    n_threads=8,
    refine_with_igraph=False,
    strict=False,
)
```

### `leiden(...)` common parameters

- `backend`: `"openmp"` (default), `"igraph"`, `"python"`, `"leidenalg"` (AnnData only, via Scanpy fallback)
- `n_threads`: OpenMP thread count (`None` means backend default)
- `refine_with_igraph`: only for `backend="openmp"`; if OpenMP succeeds, run igraph refine using OpenMP labels as initial membership
- `strict`: if `True`, raise immediately on unsupported/failure; if `False`, allow fallback path
- `return_quality`: return quality/iteration metadata

## Notes

- `backend="openmp"` uses the compiled extension (`_core`) if available.
- Fallback order for `backend="openmp"`: `igraph/leidenalg` -> `scanpy.tl.leiden` (if available) -> `backend="python"` path.
- Current `backend="python"` implementation delegates to `igraph/leidenalg`.
- `partition_type` is currently unsupported in the custom backend and requires fallback.

## Vendor source policy

`cpp/vendor/leiden_communities_openmp` is the designated location for vendored upstream C++ code from:

- <https://github.com/puzzlef/leiden-communities-openmp>

The current implementation ships a local adapter/scaffold and expects a pinned upstream commit to be vendored into that folder during integration.

## Compare accuracy and speed on AnnData

Use the script below to compare baseline Scanpy Leiden (leidenalg) vs patch-mode OpenMP backend on the same `.h5ad`:

```bash
uv run python scripts/compare_adata.py --h5ad /path/to/data.h5ad --resolution 1.0 --random-state 0 --n-threads 8
```

Notes:
- If `neighbors` is missing, the script will run `sc.pp.neighbors(...)` automatically.
- Use `--strict` to fail when OpenMP backend is unavailable instead of falling back.
- Use `--refine-with-igraph` to enable one igraph refine pass after OpenMP.
- The script saves a UMAP figure that visualizes baseline vs patch results (`--out-plot`).
