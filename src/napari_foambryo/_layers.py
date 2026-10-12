"""The plugin's layers: the mesh over the labels, tensions, pressures and crop faces; and the session's runs.

**Frames.** A Labels layer with `scale = s` and `translate = t` places voxel `j` at world `j · s + t`. The library's
mesh is in the unit of the spacing used, `p = j · h`, with the first voxel's centre at the origin. A Surface layer
with `scale = s / h` and `translate = t` therefore puts every mesh point exactly on its voxel: with the spacing taken
from the layer itself (`h = s`), the scale is 1 and no point is rescaled.

Every layer records `metadata["foambryo"]` (kind, run identifier, source layer), which Infer, Export and the
writer read. Runs are kept in this module's registry for the session.

**Explode and cut** (the View dock). A run's layers are drawn with an explode coefficient (`0` = off): above 0, the
tensions are drawn once per cell side (`interface_side_copy`) and the pressures once per cell (`cell_side_copy`), and
every vertex is moved by the library's displacement of its cell (`_display.exploded_vertices`); the copies change
only when the coefficient leaves or returns to 0, and moving the slider only moves vertices. A layer is updated in
place, so its visibility, opacity and blending stay the user's. The cut is a napari clipping plane on every layer of
the run and on its Labels layer, given in each layer's own data coordinates, so that it is one plane in the world.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from napari_foambryo._display import (
    DisplayCopy,
    cell_side_copy,
    crop_face_copy,
    exploded_vertices,
    interface_copy,
    interface_side_copy,
    pressure_copy,
    pressure_vertex_colours,
    scattered_directions,
    tension_vertex_colours,
)

if TYPE_CHECKING:
    from napari_foambryo._runs import Run

__all__ = [
    "RUNS",
    "CutPlane",
    "add_mesh_layers",
    "add_result_layers",
    "apply_cut",
    "explode_layers",
    "highlight",
    "new_run_id",
    "run_of_layer",
]

#: The session's runs, by identifier.
RUNS: dict[str, Run] = {}
_COUNTER = itertools.count(1)
#: Each run's displacement directions (`_display.scattered_directions`), computed on first use.
_DIRECTIONS: dict[str, np.ndarray] = {}

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


def _directions(run: Run) -> np.ndarray:
    if run.run_id not in _DIRECTIONS:
        _DIRECTIONS[run.run_id] = scattered_directions(run.mesh)
    return _DIRECTIONS[run.run_id]


def _set_surface(
    viewer: Any,  # noqa: ANN401
    run: Run,
    name: str,
    copy: DisplayCopy,
    colours: np.ndarray | None,
    metadata: dict,
    explode: float = 0.0,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Show a display copy as the Surface layer `name`: updated in place when it is this run's, else added.

    Updated in place, the layer keeps the user's visibility, opacity and blending; `kwargs` apply only to a new layer.
    """
    vertices = exploded_vertices(copy, _directions(run), explode) if explode else copy.vertices
    metadata["foambryo"]["explode"] = float(explode)
    if name in viewer.layers:
        layer = viewer.layers[name]
        if layer.metadata.get("foambryo", {}).get("run_id") == run.run_id and hasattr(layer, "vertex_colors"):
            layer.vertex_colors = None  # napari checks the colours against the vertices when the data changes
            layer.data = (vertices, copy.faces)
            layer.vertex_colors = colours
            layer.metadata = metadata
            return layer
        viewer.layers.remove(name)
    return viewer.add_surface((vertices, copy.faces), name=name, vertex_colors=colours, metadata=metadata, **kwargs)


def _metadata(run: Run, kind: str, copy: DisplayCopy, **extra: Any) -> dict:  # noqa: ANN401
    return {
        "foambryo": {
            "kind": kind,
            "run_id": run.run_id,
            "source_vertex": copy.source_vertex,
            "source_triangle": copy.source_triangle,
            "vertex_group": copy.vertex_group,
            "vertex_cell": copy.vertex_cell,
            "unexploded_vertices": copy.vertices,
            **extra,
        }
    }


def add_mesh_layers(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    base_name: str,
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
    *,
    explode: float = 0.0,
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
    _set_surface(viewer, run, name, cells, colours, _metadata(run, "cells", cells), explode, **frame)
    names.append(name)
    crop = crop_face_copy(run.mesh)
    if crop is not None:
        name = f"{base_name} crop faces"
        colours = np.tile(np.asarray(CROP_FACE_RGBA, dtype=np.float32), (len(crop.vertices), 1))
        _set_surface(
            viewer,
            run,
            name,
            crop,
            colours,
            _metadata(run, "crop faces", crop),
            explode,
            opacity=CROP_FACE_OPACITY,
            blending="translucent",
            **frame,
        )
        names.append(name)
    return names


def _tension_layer_content(
    run: Run, tensions: dict, pressures: dict | None, exploded: bool
) -> tuple[DisplayCopy, np.ndarray, dict]:
    copy, keys = interface_side_copy(run.mesh) if exploded else interface_copy(run.mesh)
    values, colours = tension_vertex_colours(run.mesh, copy, keys, tensions, pressures)
    return copy, colours, {"interfaces": keys, "base_colours": colours, "values": values}


def _pressure_layer_content(run: Run, tensions: dict, pressures: dict, exploded: bool) -> tuple:
    """Per higher-labelled material as `foambryo view` draws it; exploded, per cell side. One scale either way."""
    copy = cell_side_copy(run.mesh) if exploded else pressure_copy(run.mesh)
    values, colours = pressure_vertex_colours(run.mesh, copy, tensions, pressures)
    return copy, colours, {"base_colours": colours, "values": values}


def add_result_layers(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    base_name: str,
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
    *,
    explode: float = 0.0,
) -> list[str]:
    """Add `<base> tensions` and, when there are pressures, `<base> pressures`; returns the layers' names.

    Without pressures (the tensions only, or pressures the library refused) an earlier `<base> pressures` layer is
    removed, so that no pressure is shown that this inference did not compute.
    """
    outcome = run.outcome
    if outcome is None or outcome.tensions is None:
        return []
    frame = _frame(source_layer, spacing_used)
    names = []
    copy, colours, extra = _tension_layer_content(run, outcome.tensions, outcome.pressures, explode > 0)
    name = f"{base_name} tensions"
    _set_surface(viewer, run, name, copy, colours, _metadata(run, "tensions", copy, **extra), explode, **frame)
    names.append(name)
    if outcome.pressures is not None:
        copy, colours, extra = _pressure_layer_content(run, outcome.tensions, outcome.pressures, explode > 0)
        name = f"{base_name} pressures"
        _set_surface(viewer, run, name, copy, colours, _metadata(run, "pressures", copy, **extra), explode, **frame)
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
    *,
    explode: float = 0.0,
) -> str:
    """Show a seed ensemble's tensions on the run's own mesh (the reconstruction's default seed, 42).

    The ensemble has no mesh of its own (decision Q4): its tensions are matched to the displayed mesh by label
    pair; an interface of the mesh that the ensemble did not determine is grey; an ensemble interface absent from
    the mesh is only in the table. The colours are the viewer's tension scale over the ensemble's tensions.
    """
    frame = _frame(source_layer, spacing_used)
    copy, colours, extra = _tension_layer_content(run, run.ensemble.tensions, None, explode > 0)
    name = ensemble_layer_name(base_name, len(run.ensemble.seeds))
    for layer in list(viewer.layers):
        info = layer.metadata.get("foambryo", {})
        if info.get("kind") == "ensemble tensions" and layer.name.startswith(base_name) and layer.name != name:
            viewer.layers.remove(layer)
    _set_surface(viewer, run, name, copy, colours, _metadata(run, "ensemble tensions", copy, **extra), explode, **frame)
    return name


def explode_layers(run: Run, layers: list, explode: float) -> None:
    """Draw a run's layers with the explode coefficient `explode` (0 = off), in place.

    The tensions and pressures layers change copy when the coefficient leaves or returns to 0 (each interface drawn
    once, or once per cell side); otherwise only their vertices move. The colours are those of the same values on the
    same scales: the pressure scale is the unexploded surface's, so the Results dock's colour bar holds for both.
    """
    directions = _directions(run)
    for layer in layers:
        info = layer.metadata.get("foambryo", {})
        kind = info.get("kind")
        if info.get("run_id") != run.run_id or kind is None:
            continue
        now = explode > 0
        if kind in ("tensions", "pressures", "ensemble tensions") and (info.get("explode", 0.0) > 0) != now:
            if kind == "ensemble tensions":
                copy, colours, extra = _tension_layer_content(run, run.ensemble.tensions, None, now)
            elif kind == "tensions":
                copy, colours, extra = _tension_layer_content(run, run.outcome.tensions, run.outcome.pressures, now)
            else:
                copy, colours, extra = _pressure_layer_content(run, run.outcome.tensions, run.outcome.pressures, now)
            metadata = _metadata(run, kind, copy, **extra)
            metadata["foambryo"]["explode"] = float(explode)
            layer.vertex_colors = None
            layer.data = (exploded_vertices(copy, directions, explode), copy.faces)
            layer.vertex_colors = colours
            layer.metadata = metadata
            continue
        copy = DisplayCopy(
            info["unexploded_vertices"],
            layer.data[1],
            info["source_vertex"],
            info["source_triangle"],
            info["vertex_group"],
            np.zeros(0, dtype=np.int64),
            vertex_cell=info.get("vertex_cell"),
        )
        colours = layer.vertex_colors
        layer.data = (exploded_vertices(copy, directions, explode), copy.faces)
        layer.vertex_colors = colours
        layer.metadata = {"foambryo": {**info, "explode": float(explode)}}


@dataclass(frozen=True)
class CutPlane:
    """A cut normal to one axis, in world coordinates: what is kept is the side `keep` says.

    Attributes:
        axis: 0, 1 or 2 (z, y, x: napari's and the mask's order).
        position: where the plane crosses the axis, in world units (the mesh's unit over a Labels layer).
        keep: `"lower"` keeps the coordinates below `position`, `"higher"` those above.
    """

    axis: int
    position: float
    keep: str = "lower"


def apply_cut(layers: list, cut: CutPlane | None) -> None:
    """Clip each layer by the world plane `cut` (napari's clipping planes), or remove the cut (`None`).

    napari places a layer's clipping plane in its data coordinates: the plane's position along the axis is
    `(position − translate) / scale` and its normal points to the kept side (`(0, 0, −1)` keeps x below the
    position: measured in a real window, napari 0.9.2), its sign turned by a negative scale. Only the last three
    (displayed) axes are cut.
    """
    for layer in layers:
        if not hasattr(layer, "experimental_clipping_planes"):
            continue
        if cut is None:
            layer.experimental_clipping_planes = []
            continue
        scale = float(np.asarray(layer.scale)[-3:][cut.axis])
        translate = float(np.asarray(layer.translate)[-3:][cut.axis])
        position = [0.0, 0.0, 0.0]
        position[cut.axis] = (cut.position - translate) / scale
        normal = [0.0, 0.0, 0.0]
        normal[cut.axis] = (-1.0 if cut.keep == "lower" else 1.0) * (1.0 if scale > 0 else -1.0)
        layer.experimental_clipping_planes = [{"position": tuple(position), "normal": tuple(normal), "enabled": True}]


def world_bounds(layers: list) -> np.ndarray | None:
    """The `(2, 3)` world box (z, y, x) of the Surface layers among `layers`, as drawn now (exploded or not)."""
    boxes = []
    for layer in layers:
        if not hasattr(layer, "vertex_colors") or not len(layer.data[0]):
            continue
        world = np.asarray(layer.data[0])[:, -3:] * np.asarray(layer.scale)[-3:] + np.asarray(layer.translate)[-3:]
        boxes.append(np.stack([world.min(axis=0), world.max(axis=0)]))
    if not boxes:
        return None
    stacked = np.stack(boxes)
    return np.stack([stacked[:, 0].min(axis=0), stacked[:, 1].max(axis=0)])
