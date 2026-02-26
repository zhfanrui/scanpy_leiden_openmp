from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Optional

from ._api import leiden

_ORIGINAL_SCANPY_LEIDEN: Optional[Callable[..., Any]] = None
_IS_PATCHED = False


def _get_scanpy_module() -> ModuleType:
    import scanpy as sc  # type: ignore

    return sc


def patch_scanpy() -> None:
    global _ORIGINAL_SCANPY_LEIDEN
    global _IS_PATCHED

    if _IS_PATCHED:
        return

    sc = _get_scanpy_module()
    if not hasattr(sc, "tl") or not hasattr(sc.tl, "leiden"):
        raise AttributeError("scanpy.tl.leiden was not found")

    original = sc.tl.leiden

    def _patched_leiden(adata: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            if kwargs.get("strict", False):
                raise TypeError("Positional arguments are not supported by patched leiden backend")
            return original(adata, *args, **kwargs)

        backend = kwargs.pop("backend", "openmp")
        strict = kwargs.pop("strict", False)
        n_threads = kwargs.pop("n_threads", None)
        return_quality = kwargs.pop("return_quality", False)

        return leiden(
            adata,
            backend=backend,
            strict=strict,
            n_threads=n_threads,
            return_quality=return_quality,
            _scanpy_fallback=original,
            **kwargs,
        )

    _ORIGINAL_SCANPY_LEIDEN = original
    sc.tl.leiden = _patched_leiden
    _IS_PATCHED = True


def unpatch_scanpy() -> None:
    global _ORIGINAL_SCANPY_LEIDEN
    global _IS_PATCHED

    if not _IS_PATCHED:
        return

    sc = _get_scanpy_module()
    if _ORIGINAL_SCANPY_LEIDEN is not None:
        sc.tl.leiden = _ORIGINAL_SCANPY_LEIDEN
    _ORIGINAL_SCANPY_LEIDEN = None
    _IS_PATCHED = False
