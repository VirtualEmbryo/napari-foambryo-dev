"""Display copies of a multimaterial mesh, and their colours, computed by foambryo's own colour functions.

napari's Surface layer colours **vertices**, not faces; a multimaterial mesh shares every junction vertex between
three or more interfaces. A uniform colour per interface (tensions), per higher-labelled material (pressures, as
`foambryo view` draws its surface) or per cell side (cells) therefore needs a **display copy** in which each vertex
is duplicated once per group it belongs to. The library's mesh is never changed; each copy keeps the index of the
library's vertex and triangle behind each of its own, so that every displayed number maps back exactly.

Colours are those of the Polyscope viewer, computed by `foambryo.colour_scales` on the same values: `jet` between the
5 % and 95 % quantiles of the tensions, reversed `magma` from 0 to the 95 % quantile of the pressures over the
surface's triangles, grey for `nan`. No Qt and no napari here.

**The exploded view** (the Polyscope viewer's "scattered" view) moves each cell away from the tissue's centre by the
library's own displacement, `DcelData.compute_scattered_arrays(coeff)`: `coeff × (cell centroid − tissue centroid)`.
Each interface is then drawn twice, once on each of its cells' sides (`interface_side_copy`), and every copy records
in `vertex_cell` the cell that moves each of its vertices. `scattered_directions` reads the library's displacements
once, at `coeff = 1`; `exploded_vertices` multiplies them by the coefficient, which is the library's own arithmetic
(`1.0 × d` is exactly `d`), so the exploded positions are bitwise those of `compute_scattered_arrays(coeff)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from foambryo.colour_scales import (
    UNDETERMINED_COLOR,
    clipped_colour_fractions,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colours,
    tension_colours,
)
from numpy.typing import NDArray

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

__all__ = [
    "DisplayCopy",
    "cell_side_copy",
    "crop_face_copy",
    "exploded_vertices",
    "interface_copy",
    "interface_keys",
    "interface_side_copy",
    "pressure_copy",
    "pressure_vertex_colours",
    "scattered_directions",
    "tension_vertex_colours",
]


@dataclass(frozen=True)
class DisplayCopy:
    """A copy of (part of) the mesh with its vertices duplicated per group.

    Attributes:
        vertices: `(n, 3)` positions, copies of the library's (bitwise).
        faces: `(m, 3)` triangles, indices into `vertices`.
        source_vertex: `(n,)` the library vertex each copied vertex is.
        source_triangle: `(m,)` the library triangle (row of `mesh.f`) each copied triangle is.
        vertex_group: `(n,)` the group of each copied vertex (an interface index, or a label).
        face_group: `(m,)` the group of each copied triangle.
        vertex_cell: `(n,)` the cell whose displacement moves each vertex in the exploded view, or `None` for a
            copy that cannot be exploded (an interface drawn once is shared by two cells).
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    source_vertex: NDArray[np.int64]
    source_triangle: NDArray[np.int64]
    vertex_group: NDArray[np.int64]
    face_group: NDArray[np.int64]
    vertex_cell: NDArray[np.int64] | None = None

    @property
    def nbytes(self) -> int:
        """Bytes of the arrays handed to napari (vertices and faces)."""
        return int(self.vertices.nbytes + self.faces.nbytes)


def _copy_by_group(
    points: NDArray[np.float64], triangles: NDArray[np.int64], rows: NDArray[np.int64], groups: NDArray[np.int64]
) -> DisplayCopy:
    """Copy the triangles `rows` of the mesh, each in its group, with one vertex per (group, vertex) pair."""
    n_points = len(points)
    groups = np.asarray(groups, dtype=np.int64)
    corners = triangles[rows].reshape(-1)
    keys = np.repeat(groups, 3) * n_points + corners
    unique, inverse = np.unique(keys, return_inverse=True)
    source_vertex = unique % n_points
    return DisplayCopy(
        vertices=points[source_vertex],
        faces=inverse.reshape(-1, 3).astype(np.int64),
        source_vertex=source_vertex.astype(np.int64),
        source_triangle=np.asarray(rows, dtype=np.int64),
        vertex_group=(unique // n_points).astype(np.int64),
        face_group=groups,
    )


def interface_keys(mesh: DcelData) -> tuple[NDArray[np.int64], list[tuple[int, int]]]:
    """Each triangle's interface index, and the interfaces as `(label_1, label_2)` with `label_1 < label_2`."""
    pairs = np.sort(np.asarray(mesh.f[:, 3:5], dtype=np.int64), axis=1)
    unique, index = np.unique(pairs, axis=0, return_inverse=True)
    return index.reshape(-1).astype(np.int64), [(int(a), int(b)) for a, b in unique]


def interface_copy(mesh: DcelData) -> tuple[DisplayCopy, list[tuple[int, int]]]:
    """One copy per interface, crop faces left out (the library keeps them out of the equations).

    Returns:
        tuple: the copy (its groups are indices into the list), and the interfaces `(label_1, label_2)`.
    """
    index, keys = interface_keys(mesh)
    rows = np.flatnonzero(~np.asarray(mesh.crop_faces, dtype=bool))
    return _copy_by_group(
        np.asarray(mesh.v, dtype=np.float64), np.asarray(mesh.f[:, :3], dtype=np.int64), rows, index[rows]
    ), keys


def pressure_copy(mesh: DcelData) -> DisplayCopy:
    """One copy per higher-labelled material of each triangle, as `foambryo view` colours its surface."""
    rows = np.arange(len(mesh.f), dtype=np.int64)
    groups = np.max(np.asarray(mesh.f[:, 3:5], dtype=np.int64), axis=1)
    return _copy_by_group(np.asarray(mesh.v, dtype=np.float64), np.asarray(mesh.f[:, :3], dtype=np.int64), rows, groups)


def cell_side_copy(mesh: DcelData) -> DisplayCopy:
    """Each cell as a closed surface: every triangle once per cell on either side (the medium has none).

    The second side's triangles are reversed, so that each cell's triangles are wound alike.
    """
    triangles = np.asarray(mesh.f[:, :3], dtype=np.int64)
    labels = np.asarray(mesh.f[:, 3:5], dtype=np.int64)
    first = np.flatnonzero(labels[:, 0] > 0)
    second = np.flatnonzero(labels[:, 1] > 0)
    stacked = np.concatenate([triangles, triangles[:, ::-1]])
    rows = np.concatenate([first, second + len(triangles)])
    groups = np.concatenate([labels[first, 0], labels[second, 1]])
    copy = _copy_by_group(np.asarray(mesh.v, dtype=np.float64), stacked, rows, groups)
    return DisplayCopy(
        copy.vertices,
        copy.faces,
        copy.source_vertex,
        rows % len(triangles),
        copy.vertex_group,
        copy.face_group,
        vertex_cell=copy.vertex_group,
    )


def interface_side_copy(mesh: DcelData) -> tuple[DisplayCopy, list[tuple[int, int]]]:
    """Each interface once per cell on either side of it (the medium has no side), crop faces left out.

    For the exploded view: every vertex is duplicated per (cell, interface), the groups are interface indices into
    the returned list (as `interface_copy`'s, so a tension colours both sides alike and a table row selects both),
    and `vertex_cell` is the cell each side belongs to. The second side's triangles are reversed, as in
    `cell_side_copy`.

    Returns:
        tuple: the copy, and the interfaces `(label_1, label_2)`.
    """
    index, keys = interface_keys(mesh)
    triangles = np.asarray(mesh.f[:, :3], dtype=np.int64)
    labels = np.asarray(mesh.f[:, 3:5], dtype=np.int64)
    kept = ~np.asarray(mesh.crop_faces, dtype=bool)
    first = np.flatnonzero(kept & (labels[:, 0] > 0))
    second = np.flatnonzero(kept & (labels[:, 1] > 0))
    stacked = np.concatenate([triangles, triangles[:, ::-1]])
    rows = np.concatenate([first, second + len(triangles)])
    cells = np.concatenate([labels[first, 0], labels[second, 1]])
    interfaces = index[rows % len(triangles)]
    n_interfaces = max(len(keys), 1)
    copy = _copy_by_group(np.asarray(mesh.v, dtype=np.float64), stacked, rows, cells * n_interfaces + interfaces)
    return DisplayCopy(
        copy.vertices,
        copy.faces,
        copy.source_vertex,
        rows % len(triangles),
        copy.vertex_group % n_interfaces,
        copy.face_group % n_interfaces,
        vertex_cell=copy.vertex_group // n_interfaces,
    ), keys


def crop_face_copy(mesh: DcelData) -> DisplayCopy | None:
    """The crop faces alone (one group), or `None` when the mesh has none.

    A crop face is labelled `(0, c)`: `vertex_cell` is `c`, the cut cell it closes, which moves it when exploded.
    """
    rows = np.flatnonzero(np.asarray(mesh.crop_faces, dtype=bool))
    if not rows.size:
        return None
    cut_cell = np.max(np.asarray(mesh.f[rows, 3:5], dtype=np.int64), axis=1)
    copy = _copy_by_group(
        np.asarray(mesh.v, dtype=np.float64), np.asarray(mesh.f[:, :3], dtype=np.int64), rows, cut_cell
    )
    return DisplayCopy(
        copy.vertices,
        copy.faces,
        copy.source_vertex,
        copy.source_triangle,
        np.zeros(len(copy.vertices), dtype=np.int64),
        np.zeros(len(copy.faces), dtype=np.int64),
        vertex_cell=copy.vertex_group,
    )


def scattered_directions(mesh: DcelData) -> NDArray[np.float64]:
    """Each label's displacement in the library's scattered view at `coeff = 1`, as a `(max label + 1, 3)` array.

    `DcelData.compute_scattered_arrays(1.0)` is called on a shallow copy of the mesh, so the library's mesh is not
    changed. Row `c` is `1.0 × (centroid of cell c − centroid of the tissue)`; the medium's row (0) and the rows of
    labels that are not cells are zero.
    """
    import copy as shallow

    probe = shallow.copy(mesh)
    probe.compute_scattered_arrays(1.0)
    labels = np.asarray(mesh.f[:, 3:5], dtype=np.int64)
    out = np.zeros((int(labels.max(initial=0)) + 1, 3), dtype=np.float64)
    for label, displacement in probe.clusters_displacements.items():
        out[int(label)] = displacement
    return out


def exploded_vertices(copy: DisplayCopy, directions: NDArray[np.float64], coeff: float) -> NDArray[np.float64]:
    """The copy's vertices moved by `coeff ×` their cell's direction: the library's scattered positions, bitwise.

    `coeff = 0` (or a copy without `vertex_cell`) returns the vertices unmoved.
    """
    if coeff == 0 or copy.vertex_cell is None:
        return copy.vertices
    return copy.vertices + coeff * directions[copy.vertex_cell]


def _rgba(rgb: NDArray[np.float64]) -> NDArray[np.float32]:
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)


def tension_vertex_colours(
    mesh: DcelData,
    copy: DisplayCopy,
    keys: list[tuple[int, int]],
    tensions: dict[tuple[int, int], float],
    pressures: dict[int, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Each copied vertex's tension and RGBA colour, on the viewer's tension scale.

    Returns:
        tuple: the tension of each vertex (`nan` when undetermined or absent), and its colour.
    """
    per_interface = np.array([_tension_of(tensions, key) for key in keys], dtype=np.float64)
    values = per_interface[copy.vertex_group]
    ranges = colour_bar_ranges(mesh, tensions, pressures)
    if ranges.tension is None:
        return values, _rgba(np.tile(UNDETERMINED_COLOR, (len(values), 1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = tension_colours(clipped_colour_fractions(values, ranges.tension))
    return values, _rgba(rgb)


def pressure_vertex_colours(
    mesh: DcelData,
    copy: DisplayCopy,
    tensions: dict[tuple[int, int], float],
    pressures: dict[int, float],
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Each copied vertex's pressure and RGBA colour, on the viewer's one pressure scale."""
    values = np.array([pressures.get(int(g), np.nan) for g in copy.vertex_group], dtype=np.float64)
    ranges = colour_bar_ranges(mesh, tensions, pressures)
    if ranges.pressure is None:
        return values, _rgba(np.tile(UNDETERMINED_COLOR, (len(values), 1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = pressure_colours(pressure_colour_fractions(values, ranges.pressure[1]))
    return values, _rgba(rgb)


def _tension_of(tensions: dict[tuple[int, int], float], key: tuple[int, int]) -> float:
    if key in tensions:
        return float(tensions[key])
    reverse = (key[1], key[0])
    return float(tensions[reverse]) if reverse in tensions else float("nan")
