"""Tiers T2 and T3: the dock widgets on a `ViewerModel` (no canvas), driven as a user would, and the widgets' defaults.

A walk per sample: choose the Labels layer, Reconstruct (child process), Infer, read the tables, select a row,
Export. The layers are checked on the model (T2); the numbers shown are the library's (the T1 tests check them).
Every widget default equals the library signature's default.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm
from foambryo import infer_forces
from foambryo import results as tables

from napari_foambryo._layers import RUNS
from napari_foambryo._sample_data import benchmark_case_000, foam_anisotropic, foam_cropped, foam_cubic
from napari_foambryo._widgets import ExportWidget, InferWidget, ReconstructWidget, ResultsWidget

TIMEOUT_MS = 180_000


@pytest.fixture(autouse=True)
def _wait_for_background_workers(qtbot):
    """Let the widgets' background work (the mask report) finish before Qt is torn down."""
    yield
    from napari_foambryo._widgets import _LIVE_WORKERS

    qtbot.waitUntil(lambda: not _LIVE_WORKERS, timeout=TIMEOUT_MS)


@pytest.fixture
def viewer():
    from napari.components import ViewerModel

    return ViewerModel()


@pytest.fixture
def docks(qtbot, viewer):
    from napari.utils._proxies import PublicOnlyProxy

    # napari gives each plugin widget its own proxy of the viewer: the widgets must still share one session
    widgets = tuple(
        cls(PublicOnlyProxy(viewer)) for cls in (ReconstructWidget, InferWidget, ResultsWidget, ExportWidget)
    )
    for widget in widgets:
        qtbot.addWidget(widget)
    return widgets


def _add(viewer, sample) -> object:
    ((data, kwargs, _),) = sample()
    return viewer.add_labels(data, **kwargs)


def _reconstruct(qtbot, reconstruct, name: str) -> None:
    reconstruct.select_layer(name)
    assert reconstruct.run_button.isEnabled(), reconstruct.spacing_label.text()
    count = len(RUNS)
    reconstruct.start()
    qtbot.waitUntil(lambda: reconstruct._worker is None, timeout=TIMEOUT_MS)
    assert reconstruct.last_error is None, reconstruct.status.text()
    assert len(RUNS) == count + 1


def _infer(qtbot, infer) -> None:
    infer.start()
    qtbot.waitUntil(lambda: infer._worker is None, timeout=TIMEOUT_MS)


def test_widget_defaults_are_the_librarys(docks):
    reconstruct, infer, _, _ = docks
    assert (
        reconstruct.min_distance.value()
        == inspect.signature(get_default_mesh_reconstruction_algorithm).parameters["min_distance"].default
    )
    signature = inspect.signature(infer_forces).parameters
    assert tables.TENSION_METHODS[infer.tension_box.currentText()] is signature["mode_tension"].default
    assert tables.PRESSURE_METHODS[infer.pressure_box.currentText()] is signature["mode_pressure"].default
    assert float(infer.mean_tension.text()) == signature["mean_tension"].default
    assert float(infer.base_pressure.text()) == signature["base_pressure"].default
    assert infer.solver_box.currentText() == signature["solver"].default
    assert tables.DIAGNOSTICS[infer.diagnostics_box.currentText()] is signature["diagnostics"].default
    assert reconstruct.medium_box.value() == 0


def test_case_000_walk_shows_true_tensions_and_exports(qtbot, viewer, docks, tmp_path):
    reconstruct, infer, results, export = docks
    layer = _add(viewer, benchmark_case_000)
    reconstruct.select_layer(layer.name)
    assert "the sample's metadata" in reconstruct.spacing_label.text()
    _reconstruct(qtbot, reconstruct, layer.name)
    assert "benchmark case 000 cells" in viewer.layers
    _infer(qtbot, infer)
    assert infer.status.text().startswith("complete"), infer.status.text()
    for name in ("benchmark case 000 tensions", "benchmark case 000 pressures"):
        assert name in viewer.layers
        assert viewer.layers[name].vertex_colors.shape[1] == 4
    assert results.interfaces.rowCount() == len(RUNS[results.run_id].outcome.tensions)
    headers = [results.interfaces.horizontalHeaderItem(c).text() for c in range(results.interfaces.columnCount())]
    assert "true tension" in headers
    assert "median relative error" in results.summary.text()
    assert results.bars.pixmap() is not None and not results.bars.pixmap().isNull()
    results.interfaces.selectRow(0)
    faded = viewer.layers["benchmark case 000 tensions"].vertex_colors
    assert (faded[:, 3] < 1).any() and (faded[:, 3] == 1).any()
    results.interfaces.clearSelection()

    export.folder.setText(str(tmp_path / "case_000"))
    written = export.export()
    assert "run.json" in written and "mesh.rec" in written
    record = json.loads((tmp_path / "case_000" / "run.json").read_text())
    assert record["source_mask"]["spacing_from"].startswith("this layer (the sample's metadata)")
    assert record["equivalent_commands"][1].startswith("foambryo infer mesh.rec -o . --tension-method young-dupre")


def test_anisotropic_sample_says_it_is_resampled(qtbot, viewer, docks):
    reconstruct, _, _, _ = docks
    layer = _add(viewer, foam_anisotropic)
    reconstruct.select_layer(layer.name)
    assert "Resampled to cubic voxels of 0.5 um" in reconstruct.spacing_label.text()
    assert "Estimate:" in reconstruct.estimate_label.text()
    _reconstruct(qtbot, reconstruct, layer.name)
    cells = viewer.layers[f"{layer.name} cells"]
    assert tuple(cells.scale) == (1.0, 1.0, 1.0)  # spacing from the layer: the mesh is drawn as it is


def test_cropped_sample_shows_crop_faces_and_offers_only_variational(qtbot, viewer, docks):
    reconstruct, infer, results, _ = docks
    layer = _add(viewer, foam_cropped)
    _reconstruct(qtbot, reconstruct, layer.name)
    assert f"{layer.name} crop faces" in viewer.layers
    model = infer.pressure_box.model()
    enabled = {infer.pressure_box.itemText(i): model.item(i).isEnabled() for i in range(infer.pressure_box.count())}
    assert enabled == {"variational": True, "laplace": False, "weighted-laplace": False}
    _infer(qtbot, infer)
    run = RUNS[results.run_id]
    cut = {int(c) for c in run.mesh.cut_cells}
    flagged = [results.cells.item(r, 2).text() for r in range(results.cells.rowCount())]
    assert flagged.count("yes") == len(cut)
    assert "CutCellPressureWarning" in results.warnings.toPlainText()


def test_unknown_voxel_size_waits_for_confirmation(viewer, docks):
    reconstruct, _, _, _ = docks
    layer = viewer.add_labels(np.zeros((8, 8, 8), dtype=np.uint8), name="no scale")
    reconstruct.select_layer(layer.name)
    assert not reconstruct.run_button.isEnabled()
    assert "voxel size unknown" in reconstruct.spacing_label.text()
    reconstruct.confirm_voxels.setChecked(True)
    assert reconstruct.run_button.isEnabled()
    reconstruct.spacing_from.setCurrentText("typed")
    for box, value in zip(reconstruct.spacing_boxes, (2.0, 1.0, 0.5), strict=True):
        box.setValue(value)
    assert "two different" in reconstruct.spacing_label.text()
    assert not reconstruct.run_button.isEnabled()


def test_painting_the_labels_marks_the_results_out_of_date(qtbot, viewer, docks, tmp_path):
    reconstruct, infer, _, export = docks
    layer = _add(viewer, foam_cubic)
    _reconstruct(qtbot, reconstruct, layer.name)
    _infer(qtbot, infer)
    export.folder.setText(str(tmp_path / "out"))
    assert export.export_button.isEnabled()
    data = layer.data.copy()
    data[0, 0, 0] = 99
    layer.data = data
    assert not export.export_button.isEnabled()
    assert viewer.layers[f"{layer.name} cells [out of date]"]
    assert export.export() == []


def test_seed_ensemble_walk_shows_the_spread_on_the_displayed_mesh(qtbot, viewer, docks, tmp_path):
    reconstruct, infer, results, export = docks
    layer = _add(viewer, foam_cubic)
    _reconstruct(qtbot, reconstruct, layer.name)
    infer.mode_box.setCurrentText(infer.ENSEMBLE)
    assert infer.pressure_box.isHidden()
    assert "Estimate:" in infer.ensemble_estimate.text()
    infer.seeds_box.setValue(2)
    _infer(qtbot, infer)
    assert infer.status.text().startswith("seed ensemble of 2 seeds"), infer.status.text()
    names = [lay.name for lay in viewer.layers if lay.metadata.get("foambryo", {}).get("kind") == "ensemble tensions"]
    assert names == [f"{layer.name} ensemble tensions (2 seeds, shown on seed 42's mesh)"]
    outlines = [lay for lay in viewer.layers if lay.metadata.get("foambryo", {}).get("kind") == "undetermined tensions"]
    assert all(lay.metadata["foambryo"]["of"] in (names[0], f"{layer.name} tensions") for lay in outlines)
    headers = [results.interfaces.horizontalHeaderItem(c).text() for c in range(results.interfaces.columnCount())]
    assert headers[:8] == [
        "label 1",
        "label 2",
        "ensemble tension",
        "seed SD",
        "relative seed SD",
        "seeds found",
        "found by fewer than half",
        "on the displayed mesh",
    ]
    assert "lower bound on the error" in results.summary.text()
    results.interfaces.selectRow(0)
    export.folder.setText(str(tmp_path / "ensemble"))
    assert "seed-ensemble/tensions.csv" in export.export()


def test_the_memory_guard_needs_run_anyway(qtbot, viewer, docks, monkeypatch):
    import psutil

    class Tiny:
        total = 2**30
        available = 2**29

    monkeypatch.setattr(psutil, "virtual_memory", lambda: Tiny)
    reconstruct, _, _, _ = docks
    layer = _add(viewer, benchmark_case_000)
    reconstruct.select_layer(layer.name)
    assert "this machine has 1 GiB" in reconstruct.estimate_label.text()
    assert reconstruct.run_anyway.isVisibleTo(reconstruct)
    assert not reconstruct.run_button.isEnabled()
    reconstruct.run_anyway.setChecked(True)
    assert reconstruct.run_button.isEnabled()
