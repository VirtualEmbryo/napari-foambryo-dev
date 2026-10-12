"""A mesh file opened as layer data: the reader's work, without a viewer (so that it can be tested alone)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from foambryo import dcel_mesh_from_file

from napari_foambryo._display import cell_side_copy, crop_face_copy
from napari_foambryo._layers import CROP_FACE_LAYER, CROP_FACE_RGBA, RUNS, _label_palette, _metadata, new_run_id
from napari_foambryo._runs import Run

__all__ = ["mesh_layers_from_file"]


def mesh_layers_from_file(path: Path) -> list[tuple[Any, dict, str]]:
    """Read a mesh file with foambryo's reader, register it as a run, and return its layers' data."""
    mesh = dcel_mesh_from_file(path)
    run = Run(run_id=new_run_id(), mesh=mesh, rec_bytes=path.read_bytes(), rec_name=path.name)
    RUNS[run.run_id] = run
    cells = cell_side_copy(mesh)
    layers = [
        (
            (cells.vertices, cells.faces),
            {
                "name": f"{path.stem} cells",
                "vertex_colors": _label_palette(cells.vertex_group),
                "metadata": _metadata(run, "cells", cells, base_name=path.stem),
            },
            "surface",
        )
    ]
    crop = crop_face_copy(mesh)
    if crop is not None:
        layers.append(
            (
                (crop.vertices, crop.faces),
                {
                    "name": f"{path.stem} crop faces",
                    "vertex_colors": np.tile(np.asarray(CROP_FACE_RGBA, dtype=np.float32), (len(crop.vertices), 1)),
                    **CROP_FACE_LAYER,
                    "metadata": _metadata(run, "crop faces", crop, base_name=path.stem),
                },
                "surface",
            )
        )
    return layers
