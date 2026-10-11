"""Gates G7 and G8 (tier T1): the display copies map back exactly, and their colours are the library's.

* G8: every copied vertex is bitwise the library's vertex it comes from; every copied triangle has the library
  triangle's vertices, group by group; crop faces are only in their own copy; a mesh in physical units lands on
  its Labels layer's voxels through the layer frame (`scale = layer scale / spacing`, `translate`).
* G7: each vertex's colour is `foambryo.colour_scales`' colour of its value, on the range of
  `colour_bar_ranges` (the colour bars' ends).
"""

from __future__ import annotations

import numpy as np
import pytest
from foambryo import infer_forces
from foambryo.colour_scales import (
    clipped_colour_fractions,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colours,
    tension_colours,
)

from napari_foambryo._display import (
    cell_side_copy,
    crop_face_copy,
    interface_copy,
    pressure_copy,
    pressure_vertex_colours,
    tension_vertex_colours,
)
from tests.conftest import quietly


@pytest.fixture(scope="module")
def meshes(samples):
    from foambryo import reconstruct_mask

    out = {}
    for name, (mask, spacing) in samples.items():
        mesh = quietly(reconstruct_mask, mask, spacing).mesh()
        out[name] = (mesh, *quietly(infer_forces, mesh))
    return out


@pytest.mark.parametrize("name", ["cubic", "anisotropic", "cropped"])
def test_every_copy_maps_back_to_the_librarys_triangles(meshes, name):
    mesh, _, _ = meshes[name]
    triangles = np.asarray(mesh.f[:, :3])
    for copy in (interface_copy(mesh)[0], pressure_copy(mesh), cell_side_copy(mesh)):
        assert np.array_equal(copy.vertices, mesh.v[copy.source_vertex])  # bitwise: a copy, not a computation
        rebuilt = copy.source_vertex[copy.faces]
        original = triangles[copy.source_triangle]
        assert np.array_equal(np.sort(rebuilt, axis=1), np.sort(original, axis=1))
        assert np.array_equal(copy.vertex_group[copy.faces], np.repeat(copy.face_group[:, None], 3, axis=1))
    crop = np.asarray(mesh.crop_faces, dtype=bool)
    tensions_copy, _ = interface_copy(mesh)
    assert not crop[tensions_copy.source_triangle].any()
    crop_copy = crop_face_copy(mesh)
    assert (crop_copy is None) == (not crop.any())
    if crop_copy is not None:
        assert crop[crop_copy.source_triangle].all()
    cells = cell_side_copy(mesh)
    labels = np.asarray(mesh.f[:, 3:5])
    assert (cells.face_group > 0).all()
    assert np.array_equal(np.sort(np.unique(cells.face_group)), np.sort(np.unique(labels[labels > 0])))


@pytest.mark.parametrize("name", ["cubic", "anisotropic", "cropped"])
def test_vertex_colours_are_the_librarys_colours(meshes, name):
    mesh, tensions, pressures = meshes[name]
    ranges = colour_bar_ranges(mesh, tensions, pressures)
    copy, keys = interface_copy(mesh)
    values, colours = tension_vertex_colours(mesh, copy, keys, tensions, pressures)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = tension_colours(clipped_colour_fractions(values, ranges.tension))
    assert np.array_equal(colours[:, :3], expected.astype(np.float32))
    copy = pressure_copy(mesh)
    values, colours = pressure_vertex_colours(mesh, copy, tensions, pressures)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = pressure_colours(pressure_colour_fractions(values, ranges.pressure[1]))
    assert np.array_equal(colours[:, :3], expected.astype(np.float32))
    if name == "cropped":
        cut = {int(c) for c in mesh.cut_cells}
        grey = np.isin(copy.vertex_group, list(cut))
        assert np.allclose(colours[grey, :3], np.float32(0.6))


def test_the_mesh_lands_on_its_voxels_through_the_layer_frame(samples):
    from foambryo import reconstruct_mask

    from napari_foambryo._layers import _frame

    mask, spacing = samples["anisotropic"]

    class Layer:  # what _frame reads of a Labels layer
        scale = spacing
        translate = (3.0, -2.0, 7.5)

    mesh = quietly(reconstruct_mask, mask, spacing).mesh()
    frame = _frame(Layer, spacing)
    assert frame["scale"] == (1.0, 1.0, 1.0)  # the spacing came from the layer: no rescaling at all
    world = mesh.v * np.asarray(frame["scale"]) + np.asarray(frame["translate"])
    indices = (world - np.asarray(Layer.translate)) / np.asarray(spacing)
    for axis in range(3):
        assert indices[:, axis].min() >= -1.0 and indices[:, axis].max() <= mask.shape[axis]
    typed = (2.0, 1.0, 1.0)  # a typed spacing other than the layer's: the frame maps it back onto the layer
    frame = _frame(Layer, typed)
    assert frame["scale"] == tuple(float(s / t) for s, t in zip(spacing, typed, strict=True))
