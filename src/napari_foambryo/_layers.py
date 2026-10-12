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

**What has no value** is never drawn in a value's colour: an undetermined tension's interface is not filled in the
tensions layer, and `<base> undetermined tensions` draws its triangles' edges in the library's undetermined green
(a wireframe over transparent faces); crop faces are `<base> crop faces`, see-through glass with their edges drawn
(the library's crop-face colours), and are left out of the pressure layers.

**Off the GUI thread.** Building a layer's display copy and colours (`prepare_*`, numpy only, no napari) is what takes
time on a large tissue; the docks run it in a worker thread, then `install_layers` hands the arrays to napari on the
GUI thread. `add_mesh_layers`, `add_result_layers` and `add_ensemble_layer` do both at once.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from foambryo.colour_scales import CROP_FACE_COLOR, CROP_FACE_EDGE_COLOR, CROP_FACE_OPACITY, UNDETERMINED_COLOR

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
    undetermined_copy,
)

if TYPE_CHECKING:
    from napari_foambryo._runs import Run

__all__ = [
    "RUNS",
    "CutPlane",
    "PreparedLayer",
    "add_mesh_layers",
    "add_result_layers",
    "apply_cut",
    "explode_layers",
    "highlight",
    "install_layers",
    "new_run_id",
    "outline_layer_name",
    "prepare_ensemble_layers",
    "prepare_explode",
    "prepare_mesh_layers",
    "prepare_result_layers",
    "remove_stale_result_layers",
    "run_of_layer",
]

#: The session's runs, by identifier.
RUNS: dict[str, Run] = {}
_COUNTER = itertools.count(1)
#: Each run's displacement directions (`_display.scattered_directions`), computed on first use.
_DIRECTIONS: dict[str, np.ndarray] = {}

#: Crop faces are the library's glass: its fill at its opacity in the vertex colours (so that the edges, drawn by the
#: layer's wireframe, stay opaque), its edge colour.
CROP_FACE_RGBA = (*(float(c) for c in CROP_FACE_COLOR), float(CROP_FACE_OPACITY))
CROP_FACE_LAYER = {
    "opacity": 1.0,
    "blending": "translucent",
    "wireframe": {"visible": True, "color": (*(float(c) for c in CROP_FACE_EDGE_COLOR), 1.0), "width": 1.0},
}
#: An undetermined tension's interface: transparent faces (alpha 0) and its edges in the library's undetermined green.
UNDETERMINED_LAYER = {
    "opacity": 1.0,
    "blending": "translucent",
    "wireframe": {"visible": True, "color": (*(float(c) for c in UNDETERMINED_COLOR), 1.0), "width": 1.5},
}


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
    vertices: np.ndarray | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Show a display copy as the Surface layer `name`: updated in place when it is this run's, else added.

    Updated in place, the layer keeps the user's visibility, opacity and blending; `kwargs` apply only to a new layer.
    `vertices` are the positions drawn when already computed (`prepare_*`), else the copy's, exploded by `explode`.
    """
    if vertices is None:
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


@dataclass
class PreparedLayer:
    """A layer's content, computed without napari (`prepare_*`), for `install_layers`.

    Attributes:
        name: the layer's name.
        kind: `metadata["foambryo"]["kind"]`.
        copy: the display copy.
        colours: its vertex colours (RGBA), or `None`.
        extra: more metadata (the interfaces, base colours, values).
        new_layer: napari keyword arguments for a new layer only (opacity, blending, wireframe).
        vertices: the positions drawn (exploded by the run's coefficient, or the copy's own).
    """

    name: str
    kind: str
    copy: DisplayCopy
    colours: np.ndarray | None
    extra: dict = field(default_factory=dict)
    new_layer: dict = field(default_factory=dict)
    vertices: np.ndarray | None = None


def _exploded(run: Run, layer: PreparedLayer, explode: float) -> PreparedLayer:
    layer.vertices = exploded_vertices(layer.copy, _directions(run), explode) if explode else layer.copy.vertices
    layer.extra = {**layer.extra, "explode": float(explode)}
    return layer


def prepare_mesh_layers(
    run: Run, base_name: str, palette: dict[int, np.ndarray] | None = None, *, explode: float = 0.0
) -> list[PreparedLayer]:
    """`<base> cells` (each cell in its Labels colour, from `palette`) and `<base> crop faces` (glass). No napari."""
    cells = cell_side_copy(run.mesh)
    if palette is not None and len(cells.vertex_group):
        colours = np.stack([palette[int(g)] for g in cells.vertex_group])
    else:
        colours = _label_palette(cells.vertex_group)
    prepared = [PreparedLayer(f"{base_name} cells", "cells", cells, colours)]
    crop = crop_face_copy(run.mesh)
    if crop is not None:
        colours = np.tile(np.asarray(CROP_FACE_RGBA, dtype=np.float32), (len(crop.vertices), 1))
        prepared.append(PreparedLayer(f"{base_name} crop faces", "crop faces", crop, colours,
                                      new_layer=CROP_FACE_LAYER))
    return [_exploded(run, layer, explode) for layer in prepared]


def _palette_of(source_layer: Any, run: Run) -> dict[int, np.ndarray] | None:  # noqa: ANN401 - a Labels layer
    """The Labels layer's colour of each cell, read on the GUI thread (napari's colormap), or `None`."""
    if source_layer is None or not hasattr(source_layer, "get_color"):
        return None
    labels = np.unique(np.asarray(run.mesh.f[:, 3:5], dtype=np.int64))
    return {int(g): np.asarray(source_layer.get_color(int(g)), dtype=np.float32) for g in labels if g > 0}


def _tension_layers(run: Run, base_name: str, name: str, kind: str, tensions: dict, pressures: dict | None,
                    exploded: bool) -> list[PreparedLayer]:
    """A tensions layer, and its undetermined interfaces' wireframe when there are any."""
    copy, keys = interface_side_copy(run.mesh) if exploded else interface_copy(run.mesh)
    values, colours = tension_vertex_colours(run.mesh, copy, keys, tensions, pressures)
    prepared = [PreparedLayer(name, kind, copy, colours, {"interfaces": keys, "base_colours": colours,
                                                           "values": values})]
    outline = undetermined_copy(copy, values)
    if outline is not None:
        colours = np.tile(np.asarray((*UNDETERMINED_COLOR, 0.0), dtype=np.float32), (len(outline.vertices), 1))
        prepared.append(PreparedLayer(outline_layer_name(name), "undetermined tensions", outline, colours,
                                      {"interfaces": keys, "of": name}, new_layer=UNDETERMINED_LAYER))
    return prepared


def outline_layer_name(tension_layer_name: str) -> str:
    """The name of the layer that draws a tensions layer's undetermined interfaces as green outlines."""
    return f"{tension_layer_name} (undetermined: outlines)"


def _pressure_layer(run: Run, name: str, tensions: dict, pressures: dict, exploded: bool) -> PreparedLayer:
    """Per higher-labelled material as `foambryo view` draws it; exploded, per cell side. One scale either way."""
    copy = cell_side_copy(run.mesh, with_crop_faces=False) if exploded else pressure_copy(run.mesh)
    values, colours = pressure_vertex_colours(run.mesh, copy, tensions, pressures)
    return PreparedLayer(name, "pressures", copy, colours, {"base_colours": colours, "values": values})


def prepare_result_layers(run: Run, base_name: str, *, explode: float = 0.0) -> list[PreparedLayer]:
    """`<base> tensions`, `<base> undetermined tensions` (when some are) and `<base> pressures` (when inferred)."""
    outcome = run.outcome
    if outcome is None or outcome.tensions is None:
        return []
    prepared = _tension_layers(run, base_name, f"{base_name} tensions", "tensions", outcome.tensions,
                               outcome.pressures, explode > 0)
    if outcome.pressures is not None:
        prepared.append(_pressure_layer(run, f"{base_name} pressures", outcome.tensions, outcome.pressures,
                                        explode > 0))
    return [_exploded(run, layer, explode) for layer in prepared]


def prepare_ensemble_layers(run: Run, base_name: str, *, explode: float = 0.0) -> list[PreparedLayer]:
    """A seed ensemble's tensions on the run's own mesh, and the wireframe of the interfaces it did not determine."""
    name = ensemble_layer_name(base_name, len(run.ensemble.seeds))
    prepared = _tension_layers(run, base_name, name, "ensemble tensions", run.ensemble.tensions, None, explode > 0)
    return [_exploded(run, layer, explode) for layer in prepared]


def install_layers(
    viewer: Any,  # noqa: ANN401 - a napari viewer
    run: Run,
    prepared: list[PreparedLayer],
    source_layer: Any = None,  # noqa: ANN401
    spacing_used: tuple[float, float, float] | None = None,
) -> list[str]:
    """Show prepared layers in napari (on the GUI thread): each updated in place when it is this run's, else added."""
    frame = _frame(source_layer, spacing_used)
    names = []
    drawn = {layer.name for layer in prepared}
    for layer in prepared:  # a tensions layer drawn now without undetermined interfaces loses its earlier outlines
        outline = outline_layer_name(layer.name)
        if layer.kind in ("tensions", "ensemble tensions") and outline not in drawn and outline in viewer.layers:
            viewer.layers.remove(outline)
    for layer in prepared:
        metadata = _metadata(run, layer.kind, layer.copy, **{k: v for k, v in layer.extra.items() if k != "explode"})
        _set_surface(viewer, run, layer.name, layer.copy, layer.colours, metadata, layer.extra.get("explode", 0.0),
                     vertices=layer.vertices, **layer.new_layer, **frame)
        names.append(layer.name)
    return names


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
    prepared = prepare_mesh_layers(run, base_name, _palette_of(source_layer, run), explode=explode)
    return install_layers(viewer, run, prepared, source_layer, spacing_used)


def remove_stale_result_layers(viewer: Any, base_name: str, names: list[str]) -> None:  # noqa: ANN401
    """Remove the pressures layer when this inference drew none: no pressure is shown that it did not compute."""
    stale = f"{base_name} pressures"
    if stale not in names and stale in viewer.layers:
        viewer.layers.remove(stale)


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
    removed, so that no pressure is shown that this inference did not compute; likewise the outlines of undetermined
    tensions this inference does not have.
    """
    names = install_layers(viewer, run, prepare_result_layers(run, base_name, explode=explode), source_layer,
                           spacing_used)
    if names:
        remove_stale_result_layers(viewer, base_name, names)
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
    faded[~selected, 3] = np.minimum(0.25, base[~selected, 3])  # an unfilled (undetermined) interface stays unfilled
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
    prepared: list[PreparedLayer] | None = None,
) -> str:
    """Show a seed ensemble's tensions on the run's own mesh (the reconstruction's default seed, 42).

    The ensemble has no mesh of its own (decision Q4): its tensions are matched to the displayed mesh by label
    pair; an interface of the mesh that the ensemble did not determine is drawn as undetermined (no fill, a green
    wireframe); an ensemble interface absent from the mesh is only in the table. The colours are the viewer's tension
    scale over the ensemble's tensions. `prepared` is `prepare_ensemble_layers`' result when computed already.
    """
    prepared = prepared if prepared is not None else prepare_ensemble_layers(run, base_name, explode=explode)
    name = prepared[0].name
    for layer in list(viewer.layers):
        info = layer.metadata.get("foambryo", {})
        if info.get("kind") == "ensemble tensions" and layer.name.startswith(base_name) and layer.name != name:
            viewer.layers.remove(layer)
            if outline_layer_name(layer.name) in viewer.layers:
                viewer.layers.remove(outline_layer_name(layer.name))
    install_layers(viewer, run, prepared, source_layer, spacing_used)
    return name


def prepare_explode(
    run: Run, base_name: str, targets: list[tuple[str, str, float]], explode: float
) -> dict[str, PreparedLayer]:
    """The new content of the layers whose display copy changes at `explode` (it leaves or returns to 0). No napari.

    Args:
        run: the run.
        base_name: the run's layer names' stem.
        targets: `(layer name, kind, its current explode)` of the run's layers, read on the GUI thread.
        explode: the new coefficient.

    Returns:
        dict: the prepared content by layer name (the tensions' undetermined wireframe with its tensions).
    """
    now = explode > 0
    prepared: dict[str, PreparedLayer] = {}
    for name, kind, current in targets:
        if (current > 0) == now or kind not in ("tensions", "pressures", "ensemble tensions"):
            continue
        if kind == "ensemble tensions":
            layers = _tension_layers(run, base_name, name, kind, run.ensemble.tensions, None, now)
        elif kind == "tensions":
            layers = _tension_layers(run, base_name, name, kind, run.outcome.tensions, run.outcome.pressures, now)
        else:
            layers = [_pressure_layer(run, name, run.outcome.tensions, run.outcome.pressures, now)]
        for layer in layers:
            prepared[layer.name] = _exploded(run, layer, explode)
    return prepared


def explode_layers(
    run: Run, layers: list, explode: float, prepared: dict[str, PreparedLayer] | None = None, base_name: str = ""
) -> None:
    """Draw a run's layers with the explode coefficient `explode` (0 = off), in place.

    The tensions and pressures layers (and the undetermined wireframe) change copy when the coefficient leaves or
    returns to 0 (each interface drawn once, or once per cell side): their new content is `prepared`
    (`prepare_explode`, computed here when not given); otherwise only their vertices move. The colours are those of
    the same values on the same scales: the pressure scale is the unexploded surface's, so the Results dock's colour
    bar holds for both.
    """
    directions = _directions(run)
    if prepared is None:
        targets = [(layer.name, layer.metadata.get("foambryo", {}).get("kind", ""),
                    layer.metadata.get("foambryo", {}).get("explode", 0.0)) for layer in layers]
        prepared = prepare_explode(run, base_name or _base_name_of(layers), targets, explode)
    for layer in layers:
        info = layer.metadata.get("foambryo", {})
        kind = info.get("kind")
        if info.get("run_id") != run.run_id or kind is None:
            continue
        if layer.name in prepared:
            new = prepared[layer.name]
            metadata = _metadata(run, new.kind, new.copy, **{k: v for k, v in new.extra.items() if k != "explode"})
            metadata["foambryo"]["explode"] = float(explode)
            layer.vertex_colors = None
            layer.data = (new.vertices, new.copy.faces)
            layer.vertex_colors = new.colours
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


def _base_name_of(layers: list) -> str:
    """The run's layer names' stem, from its tensions or cells layer."""
    for layer in layers:
        for suffix in (" tensions", " cells"):
            kind = layer.metadata.get("foambryo", {}).get("kind")
            if kind in ("tensions", "cells") and layer.name.endswith(suffix):
                return layer.name[: -len(suffix)]
    return ""


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
