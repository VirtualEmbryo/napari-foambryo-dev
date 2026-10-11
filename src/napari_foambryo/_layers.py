"""The plugin's layers: the mesh over the labels, tensions, pressures and crop faces; and the session's runs.

**Frames.** A Labels layer with `scale = s` and `translate = t` places voxel `j` at world `j · s + t`. The library's
mesh is in the unit of the spacing used, `p = j · h`, with the first voxel's centre at the origin. A Surface layer
with `scale = s / h` and `translate = t` therefore puts every mesh point exactly on its voxel: with the spacing taken
from the layer itself (`h = s`), the scale is 1 and no point is rescaled.

Every layer records `metadata["foambryo"]` (kind, run identifier, source layer), which Infer, Export and the
writer read. Runs are kept in this module's registry for the session.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from napari_foambryo._display import (
    DisplayCopy,
    cell_side_copy,
    crop_face_copy,
    interface_copy,
    pressure_copy,
    pressure_vertex_colours,
    tension_vertex_colours,
)

if TYPE_CHECKING:
    from napari_foambryo._runs import Run

__all__ = ["RUNS", "add_mesh_layers", "add_result_layers", "highlight", "new_run_id", "run_of_layer"]

#: The session's runs, by identifier.
RUNS: dict[str, Run] = {}
_COUNTER = itertools.count(1)

#: The grey of crop faces, and their opacity.
CROP_FACE_RGBA = (0.6, 0.6, 0.6, 1.0)
CROP_FACE_OPACITY = 0.35


def new_run_id() -> str:
    """A short identifier, unique in the session."""
    return f"run{next(_COUNTER)}"


def run_of_layer(layer: Any) -> Run | None:  # noqa: ANN401 - a napari layer
    """The run a plugin layer belongs to, or `None` for any other layer."""
    info = getattr(layer, "metadata", {}).get("foambryo", {})
    return RUNS.get(info.get("run_id", ""))


def _frame(source_layer: Any, spacing_used: tuple[float, float, float] | None) -> dict[str, Any]:  # noqa: ANN401
    if source_layer is None or spacing_used is None:
        return {"scale": (1.0, 1.0, 1.0), "translate": (0.0, 0.0, 0.0)}
    scale = np.asarray(source_layer.scale[-3:], dtype=np.float64) / np.asarray(spacing_used, dtype=np.float64)
    return {"scale": tuple(float(v) for v in scale), "translate": tuple(float(v) for v in source_layer.translate[-3:])}


def _replace(viewer: Any, name: str, data: tuple, **kwargs: Any) -> Any:  # noqa: ANN401
    if name in viewer.layers:
        viewer.layers.remove(name)
    return viewer.add_surface(data, name=name, **kwargs)


def _metadata(run: Run, kind: str, copy: DisplayCopy, **extra: Any) -> dict:  # noqa: ANN401
    return {
        "foambryo": {
            "kind": kind,
            "run_id": run.run_id,
            "source_vertex": copy.source_vertex,
            "source_triangle": copy.source_triangle,
            "vertex_group": copy.vertex_group,
            **extra,
        }
    }


def add_mesh_layers(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    base_name: str,
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
) -> list[str]:
    """Add `<base> cells` (each cell in its Labels colour) and `<base> crop faces`; returns the layers' names."""
    frame = _frame(source_layer, spacing_used)
    names = []
    cells = cell_side_copy(run.mesh)
    if source_layer is not None and hasattr(source_layer, "get_color"):
        palette = {
            int(g): np.asarray(source_layer.get_color(int(g)), dtype=np.float32) for g in np.unique(cells.vertex_group)
        }
        colours = np.stack([palette[int(g)] for g in cells.vertex_group]) if len(cells.vertex_group) else None
    else:
        colours = _label_palette(cells.vertex_group)
    name = f"{base_name} cells"
    _replace(
        viewer,
        name,
        (cells.vertices, cells.faces),
        vertex_colors=colours,
        metadata=_metadata(run, "cells", cells),
        **frame,
    )
    names.append(name)
    crop = crop_face_copy(run.mesh)
    if crop is not None:
        name = f"{base_name} crop faces"
        colours = np.tile(np.asarray(CROP_FACE_RGBA, dtype=np.float32), (len(crop.vertices), 1))
        _replace(
            viewer,
            name,
            (crop.vertices, crop.faces),
            vertex_colors=colours,
            opacity=CROP_FACE_OPACITY,
            blending="translucent",
            metadata=_metadata(run, "crop faces", crop),
            **frame,
        )
        names.append(name)
    return names


def add_result_layers(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    base_name: str,
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
) -> list[str]:
    """Add `<base> tensions` and, when there are pressures, `<base> pressures`; returns the layers' names."""
    outcome = run.outcome
    if outcome is None or outcome.tensions is None:
        return []
    frame = _frame(source_layer, spacing_used)
    names = []
    copy, keys = interface_copy(run.mesh)
    values, colours = tension_vertex_colours(run.mesh, copy, keys, outcome.tensions, outcome.pressures)
    name = f"{base_name} tensions"
    _replace(
        viewer,
        name,
        (copy.vertices, copy.faces),
        vertex_colors=colours,
        metadata=_metadata(run, "tensions", copy, interfaces=keys, base_colours=colours, values=values),
        **frame,
    )
    names.append(name)
    if outcome.pressures is not None:
        copy = pressure_copy(run.mesh)
        values, colours = pressure_vertex_colours(run.mesh, copy, outcome.tensions, outcome.pressures)
        name = f"{base_name} pressures"
        _replace(
            viewer,
            name,
            (copy.vertices, copy.faces),
            vertex_colors=colours,
            metadata=_metadata(run, "pressures", copy, base_colours=colours, values=values),
            **frame,
        )
        names.append(name)
    elif f"{base_name} pressures" in viewer.layers:
        viewer.layers.remove(f"{base_name} pressures")
    return names


def highlight(viewer: Any, layer: Any, group: int | None) -> None:  # noqa: ANN401
    """Show one group (an interface index or a label) in its colours, the others faded; `None` restores all.

    The camera is centred on the group.
    """
    info = layer.metadata["foambryo"]
    base = info["base_colours"]
    if group is None:
        layer.vertex_colors = base
        return
    selected = info["vertex_group"] == group
    faded = base.copy()
    faded[~selected, :3] = 0.85 + 0.15 * base[~selected, :3]
    faded[~selected, 3] = 0.25
    layer.vertex_colors = faded
    if selected.any():
        centre = layer.data[0][selected].mean(axis=0)
        world = centre * np.asarray(layer.scale) + np.asarray(layer.translate)
        viewer.camera.center = tuple(float(v) for v in world)


def _label_palette(groups: np.ndarray) -> np.ndarray:
    """Distinct colours per label when no Labels layer gives them (a mesh opened from a file)."""
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    return np.asarray(cmap((np.asarray(groups) % 20) / 19.0), dtype=np.float32)


def ensemble_layer_name(base_name: str, n_seeds: int) -> str:
    """The ensemble layer's name says what it shows, and on which mesh."""
    return f"{base_name} ensemble tensions ({n_seeds} seeds, shown on seed 42's mesh)"


def add_ensemble_layer(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    base_name: str,
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
) -> str:
    """Show a seed ensemble's tensions on the run's own mesh (the reconstruction's default seed, 42).

    The ensemble has no mesh of its own (decision Q4): its tensions are matched to the displayed mesh by label
    pair; an interface of the mesh that the ensemble did not determine is grey; an ensemble interface absent from
    the mesh is only in the table. The colours are the viewer's tension scale over the ensemble's tensions.
    """
    frame = _frame(source_layer, spacing_used)
    result = run.ensemble
    copy, keys = interface_copy(run.mesh)
    values, colours = tension_vertex_colours(run.mesh, copy, keys, result.tensions, None)
    name = ensemble_layer_name(base_name, len(result.seeds))
    for layer in list(viewer.layers):
        if layer.metadata.get("foambryo", {}).get("kind") == "ensemble tensions" and layer.name.startswith(base_name):
            viewer.layers.remove(layer)
    _replace(
        viewer,
        name,
        (copy.vertices, copy.faces),
        vertex_colors=colours,
        metadata=_metadata(run, "ensemble tensions", copy, interfaces=keys, base_colours=colours, values=values),
        **frame,
    )
    return name
