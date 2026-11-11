"""TESSERA-IDR: GIN (grammars of IDRs) assignment of intrinsically disordered
regions, on whole sequences or as multi-block sliding windows, built on the
NARDINI+ feature pipeline.

Modules
-------
tesseraidr.multiblock   sliding-window / multi-block GIN assignment
tesseraidr.fullseq      full-sequence GIN assignment
tesseraidr.summary      out_summary -> block table, domain table, plots

Submodules are imported lazily, so ``import tesseraidr`` stays fast.
"""

from importlib import import_module
from pathlib import Path

__version__ = "1.0.0"

DATA_DIR = Path(__file__).resolve().parent / "data"
"""Folder holding the bundled reference tables (GIN centroids, compositional
feature normalisation, GIN definitions, human seed thresholds)."""

__all__ = ["multiblock", "fullseq", "summary", "DATA_DIR", "data_path", "__version__"]


def data_path(name: str) -> Path:
    """Absolute path of a bundled reference file, e.g.
    ``data_path("All_sapiens_IDRs_GIN_centroids.csv.gz")``."""
    path = DATA_DIR / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in DATA_DIR.iterdir()))
        raise FileNotFoundError(f"No bundled file {name!r}. Available: {available}")
    return path


def __getattr__(name):
    if name in ("multiblock", "fullseq", "summary"):
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
