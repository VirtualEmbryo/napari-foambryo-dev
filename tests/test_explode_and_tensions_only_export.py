"""Tier T1: the exploded view is the library's scattered view, and the tensions-only and diagnostics exports are the
command's files.

* **Explode.** Every exploded vertex (cells, interfaces drawn once per cell side, crop faces) is bitwise at the
  position `DcelData.compute_scattered_arrays(coeff)` gives that vertex in that cell, for two coefficients: the plugin
  reads the library's displacement once and scales it, which must not change a bit. Each interface's two sides map
  back to the library's triangles and carry the interface's own tension colour.
* **Tensions only** (`InferenceOptions(pressures=False)`): Export writes no `pressures.csv`, its `run.json` has the
  values `foambryo infer --no-pressures` records, and its equivalent command lines reproduce its tables byte for byte.
* **Diagnostics.** With `basic` or `spectrum`, Export writes `solves.csv` (equal to the command's but for the solves'
  wall times) and, with the spectrum, `spectra.npz` (equal to the command's, array for array).
* **Replacing an earlier export** deletes only foambryo's own outputs, never another file in the folder.
"""

from __future__ import annotations

import copy as shallow
import json
import os
import shlex

import numpy as np
import pytest
from foambryo import cli
from foambryo import results as tables
from foambryo.dcel import separate_faces_dict_keep_idx

from napari_foambryo._display import (
    cell_side_copy,
    crop_face_copy,
    exploded_vertices,
    interface_copy,
    interface_keys,
    interface_side_copy,
    scattered_directions,
    tension_vertex_colours,
)
from napari_foambryo._runs import InferenceOptions, earlier_outputs, export_run
from tests.conftest import quietly
from tests.test_export_is_the_command import SAME_VALUES, _run_for

#: The columns of `solves.csv` that are wall times, and so differ from one run to the next.
TIMES = ("seconds", "spectrum_seconds")


@pytest.fixture(scope="module")
def meshes(samples):
    from foambryo import infer_forces, reconstruct_mask

    out = {}
    for name in ("cubic", "cropped"):
        mask, spacing = samples[name]
        mesh = quietly(reconstruct_mask, mask, spacing).mesh()
        out[name] = (mesh, *quietly(infer_forces, mesh))
    return out


def _library_scattered_positions(mesh, coeff: float) -> tuple[np.ndarray, np.ndarray]:
    """`compute_scattered_arrays(coeff)`'s vertices, keyed by `cell × n_points + library vertex`, sorted by key."""
    probe = shallow.copy(mesh)
    probe.compute_scattered_arrays(coeff)
    _, clusters_idx = separate_faces_dict_keep_idx(np.asarray(mesh.f))
    keys = [
        key * len(mesh.v) + np.unique(np.asarray(mesh.f)[clusters_idx[key], :3])
        for key in sorted(clusters_idx)
        if key != 0
    ]
    keys = np.concatenate(keys)
    order = np.argsort(keys)
    return keys[order], np.asarray(probe.v_scattered)[order]


@pytest.mark.parametrize("name", ["cubic", "cropped"])
@pytest.mark.parametrize("coeff", [0.2, 0.55])
def test_exploded_positions_are_the_librarys_scattered_positions(meshes, name, coeff):
    mesh = meshes[name][0]
    keys, positions = _library_scattered_positions(mesh, coeff)
    directions = scattered_directions(mesh)
    copies = [cell_side_copy(mesh), interface_side_copy(mesh)[0]]
    if crop_face_copy(mesh) is not None:
        copies.append(crop_face_copy(mesh))
    for copy in copies:
        exploded = exploded_vertices(copy, directions, coeff)
        wanted = copy.vertex_cell * len(mesh.v) + copy.source_vertex
        found = np.searchsorted(keys, wanted)
        assert np.array_equal(keys[found], wanted)  # every exploded vertex is one the library moves
        assert exploded.tobytes() == positions[found].tobytes()  # bitwise
    assert exploded_vertices(copies[0], directions, 0.0) is copies[0].vertices  # 0 = off: nothing moves


def test_scattered_directions_leave_the_librarys_mesh_unchanged(meshes):
    mesh = meshes["cubic"][0]
    before = (mesh.v.tobytes(), np.asarray(mesh.f).tobytes())
    scattered_directions(mesh)
    assert (mesh.v.tobytes(), np.asarray(mesh.f).tobytes()) == before
    assert not hasattr(mesh, "v_scattered")


@pytest.mark.parametrize("name", ["cubic", "cropped"])
def test_each_interface_side_maps_back_and_has_the_interfaces_colour(meshes, name):
    mesh, tensions, pressures = meshes[name]
    sides, keys = interface_side_copy(mesh)
    assert keys == interface_keys(mesh)[1]
    triangles, labels = np.asarray(mesh.f[:, :3]), np.asarray(mesh.f[:, 3:5])
    assert np.array_equal(sides.vertices, mesh.v[sides.source_vertex])
    assert np.array_equal(
        np.sort(sides.source_vertex[sides.faces], axis=1), np.sort(triangles[sides.source_triangle], axis=1)
    )
    assert not np.asarray(mesh.crop_faces, dtype=bool)[sides.source_triangle].any()
    index = interface_keys(mesh)[0]
    assert np.array_equal(sides.face_group, index[sides.source_triangle])
    face_cell = sides.vertex_cell[sides.faces[:, 0]]
    assert np.all((face_cell == labels[sides.source_triangle, 0]) | (face_cell == labels[sides.source_triangle, 1]))
    assert (face_cell > 0).all()  # the medium has no side
    once, _ = interface_copy(mesh)
    shared = np.isin(np.arange(len(triangles)), once.source_triangle)
    cells_per_triangle = (labels[shared] > 0).sum(axis=1)
    assert len(sides.faces) == int(cells_per_triangle.sum())  # once per cell side
    _, colours = tension_vertex_colours(mesh, sides, keys, tensions, pressures)
    _, colours_once = tension_vertex_colours(mesh, once, keys, tensions, pressures)
    by_interface = {int(g): colours_once[i].tobytes() for i, g in enumerate(once.vertex_group)}
    assert all(colours[i].tobytes() == by_interface[int(g)] for i, g in enumerate(sides.vertex_group))


def _command_rows(path):
    return [{k: v for k, v in row.items() if k not in TIMES} for row in tables.read_solve_table(path)]


def _run_the_equivalent_commands(record, exported, folder, capsys):
    folder.mkdir()
    (folder / "mask.npz").write_bytes((exported / "mask.npz").read_bytes())
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        for line in record["equivalent_commands"]:
            assert cli.main(shlex.split(line)[1:]) == cli.EXIT_OK, line
    finally:
        os.chdir(cwd)
    capsys.readouterr()


@pytest.mark.parametrize("diagnostics", ["basic", "spectrum"])
def test_tensions_only_export_is_what_foambryo_infer_no_pressures_writes(samples, tmp_path, capsys, diagnostics):
    mask, spacing = samples["cropped"]
    options = InferenceOptions("young-dupre", "variational", 1.0, 0.0, "auto", diagnostics, pressures=False)
    run = _run_for(mask, spacing, options)
    assert run.outcome.pressures is None
    exported = tmp_path / "exported"
    written = export_run(run, exported, include_mask=True)
    assert "pressures.csv" not in written and not (exported / "pressures.csv").exists()
    assert tables.SOLVE_TABLE in written
    assert (tables.SPECTRA in written) == (diagnostics == "spectrum")
    record = json.loads((exported / "run.json").read_text())
    assert "--no-pressures" in record["equivalent_commands"][1]
    assert "--pressure-method" not in record["equivalent_commands"][1]
    command = tmp_path / "command"
    _run_the_equivalent_commands(record, exported, command, capsys)
    assert (command / "tensions.csv").read_bytes() == (exported / "tensions.csv").read_bytes()
    command_record = json.loads((command / "run.json").read_text())
    for key in SAME_VALUES:
        assert record[key] == command_record[key], key
    assert record["options"]["no_pressures"] is True
    assert _command_rows(exported / tables.SOLVE_TABLE) == _command_rows(command / tables.SOLVE_TABLE)
    if diagnostics == "spectrum":
        with np.load(exported / tables.SPECTRA) as ours, np.load(command / tables.SPECTRA) as theirs:
            assert sorted(ours.files) == sorted(theirs.files) and ours.files
            for key in ours.files:
                assert np.array_equal(ours[key], theirs[key])


def test_with_pressures_the_diagnostics_export_is_the_commands(samples, tmp_path, capsys):
    mask, spacing = samples["cubic"]
    run = _run_for(mask, spacing, InferenceOptions("young-dupre", "variational", 1.0, 0.0, "auto", "spectrum"))
    exported = tmp_path / "exported"
    written = export_run(run, exported, include_mask=True)
    assert {"pressures.csv", tables.SOLVE_TABLE, tables.SPECTRA} <= set(written)
    record = json.loads((exported / "run.json").read_text())
    assert len(record["solves"]) == len(run.outcome.records) > 0
    command = tmp_path / "command"
    _run_the_equivalent_commands(record, exported, command, capsys)
    assert _command_rows(exported / tables.SOLVE_TABLE) == _command_rows(command / tables.SOLVE_TABLE)
    for table in ("tensions.csv", "pressures.csv"):
        assert (command / table).read_bytes() == (exported / table).read_bytes(), table


def test_replacing_an_earlier_export_touches_only_foambryos_files(samples, tmp_path):
    mask, spacing = samples["cubic"]
    run = _run_for(mask, spacing, InferenceOptions("young-dupre", "variational", 1.0, 0.0, "auto", "basic"))
    folder = tmp_path / "out"
    first = export_run(run, folder)
    (folder / "notes.txt").write_text("mine")
    (folder / "seed-ensemble").mkdir()
    (folder / "seed-ensemble" / "tensions.csv").write_text("an earlier ensemble's\n")
    (folder / "seed-ensemble" / "my_plot.png").write_bytes(b"not foambryo's")
    found = earlier_outputs(folder)
    assert set(found) == {*first, "seed-ensemble/tensions.csv"}
    with pytest.raises(FileExistsError):
        export_run(run, folder)
    export_run(run, folder, overwrite=True)
    assert (folder / "notes.txt").read_text() == "mine"
    assert (folder / "seed-ensemble" / "my_plot.png").read_bytes() == b"not foambryo's"
    assert not (folder / "seed-ensemble" / "tensions.csv").exists()  # an earlier output: replaced, not kept stale
