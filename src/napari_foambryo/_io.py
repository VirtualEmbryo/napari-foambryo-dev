"""The `.rec` reader and writer contributions.

* **Reader**: a `.rec` / `.arec` file (as `foambryo mesh` or delaunay-watershed-3d writes it) opens as the mesh's
  layers, `<file> cells` and, when the file records crop faces, `<file> crop faces`. The mesh is read by foambryo's
  own `dcel_mesh_from_file`, so Infer computes on it exactly what `foambryo infer` computes on the file. A file the
  reader cannot use is left to other readers (`None`).
* **Writer**: a plugin Surface layer saved as `.rec` writes its run's mesh file, byte for byte (the reconstruction's
  `.rec` bytes, or the opened file's). Any other layer is declined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from napari_foambryo._cells_from_file import mesh_layers_from_file

__all__ = ["get_reader", "write_rec"]


def get_reader(path: str | list[str]) -> Any:  # noqa: ANN401 - napari's reader protocol
    """Return the reader for one `.rec` / `.arec` path, else `None`."""
    if isinstance(path, list):
        if len(path) != 1:
            return None
        path = path[0]
    if Path(path).suffix.lower() not in (".rec", ".arec"):
        return None
    return _read


def _read(path: str | list[str]) -> list[tuple[Any, dict, str]]:
    path = path[0] if isinstance(path, list) else path
    return mesh_layers_from_file(Path(path))


def write_rec(path: str, data: Any, attributes: dict) -> list[str]:  # noqa: ANN401 - napari's writer protocol
    """Write a plugin Surface layer's mesh file; declines (returns no path) for any other layer."""
    from napari_foambryo._layers import RUNS

    info = attributes.get("metadata", {}).get("foambryo", {})
    run = RUNS.get(info.get("run_id", ""))
    if run is None or not run.rec_bytes:
        return []
    target = Path(path)
    if target.suffix.lower() not in (".rec", ".arec"):
        target = target.with_suffix(".rec")
    target.write_bytes(run.rec_bytes)
    return [str(target)]
