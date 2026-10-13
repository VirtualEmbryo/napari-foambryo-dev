"""Tiers T2 and T3: the View dock (explode, cut), the tensions only, the Diagnostics tab and Export's dialog.

Driven as a user would, on a `ViewerModel` under Qt's offscreen platform:

* **Explode** redraws the run's layers in place: exploded, the tensions are each interface once per cell side at the
  library's scattered positions; a selected row stays selected; back to 0, the layers are bitwise the unexploded
  copies again.
* **Cut** puts one enabled clipping plane on every layer of the run and on its Labels layer, at the same world
  position in each layer's own data frame (the Labels layer's voxels and the mesh's units differ on an anisotropic
  sample); off, no plane is left.
* **Tensions only**: no pressures layer, no pressure in the Cells tab, Export without `pressures.csv`, and the
  equivalent command with `--no-pressures`.
* **Diagnostics**: the tab lists one row per solve, plots the singular values with the spectrum, and Export writes
  `solves.csv` and `spectra.npz`; when the Infer dock's options changed after Infer, Export says so.
* **Export's dialog** (the real `QMessageBox`, clicked by a timer): *Cancel* writes nothing; *Replace* replaces
  foambryo's files and leaves another file alone.
* **Background**: the canvas takes the library's named colours (`BACKGROUND_COLOURS`, the Polyscope window's), a
  custom colour from the dialog (unchanged when it is cancelled), or napari's theme again; the layers' colours never
  change with it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from foambryo import results as tables
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QMessageBox

from napari_foambryo._display import exploded_vertices, interface_copy, interface_side_copy, scattered_directions
from napari_foambryo._layers import RUNS
from napari_foambryo._sample_data import foam_anisotropic, foam_cubic
from napari_foambryo._widgets import ExportWidget, InferWidget, ReconstructWidget, ResultsWidget, ViewWidget
from tests.test_widgets_walk_through import TIMEOUT_MS, _add, _infer, _reconstruct


@pytest.fixture(autouse=True)
def _wait_for_background_workers(qtbot):
    yield
    from napari_foambryo._widgets import _LIVE_WORKERS

    qtbot.waitUntil(lambda: not _LIVE_WORKERS, timeout=TIMEOUT_MS)


@pytest.fixture
def viewer():
    from napari.components import ViewerModel

    model = ViewerModel()
    model.dims.ndisplay = 3
    return model


@pytest.fixture
def docks(qtbot, viewer):
    from napari.utils._proxies import PublicOnlyProxy

    widgets = tuple(
        cls(PublicOnlyProxy(viewer))
        for cls in (ReconstructWidget, InferWidget, ResultsWidget, ExportWidget, ViewWidget)
    )
    for widget in widgets:
        qtbot.addWidget(widget)
    return widgets


def _reconstructed_and_inferred(qtbot, viewer, docks, sample=foam_cubic):
    reconstruct, infer, *_ = docks
    layer = _add(viewer, sample)
    _reconstruct(qtbot, reconstruct, layer.name)
    _infer(qtbot, infer)
    assert infer.status.text().startswith("complete"), infer.status.text()
    return layer


def test_explode_redraws_in_place_keeps_the_selection_and_returns_bitwise(qtbot, viewer, docks):
    _, _, results, _, view = docks
    layer = _reconstructed_and_inferred(qtbot, viewer, docks)
    run = RUNS[results.run_id]
    tensions = viewer.layers[f"{layer.name} tensions"]
    unexploded = (tensions.data[0].copy(), tensions.data[1].copy())
    tensions.opacity = 0.7  # the user's choice survives a redraw
    results.interfaces.selectRow(0)
    view.explode_slider.setValue(30)
    qtbot.waitUntil(
        lambda: (
            view.explode_value.text() == "0.30"
            and run.run_id in view.shared.displays
            and view.shared.displays[run.run_id].explode == 0.3
        ),
        timeout=5_000,
    )
    sides, _ = interface_side_copy(run.mesh)
    assert tensions in viewer.layers and tensions.opacity == 0.7
    assert tensions.data[0].tobytes() == exploded_vertices(sides, scattered_directions(run.mesh), 0.3).tobytes()
    assert np.array_equal(tensions.data[1], sides.faces)
    assert tensions.vertex_colors.shape == (len(sides.vertices), 4)
    assert (tensions.vertex_colors[:, 3] < 1).any()  # the selected row's interface is still the only one in colour
    cells = viewer.layers[f"{layer.name} cells"]
    assert not np.array_equal(cells.data[0], cells.metadata["foambryo"]["unexploded_vertices"])
    view.set_explode(0.45)
    assert view.explode_slider.value() == 45 and view.explode_value.text() == "0.45"  # the slider follows Python
    view.set_explode(0.0)
    assert view.explode_value.text() == "off"
    once, _ = interface_copy(run.mesh)
    assert tensions.data[0].tobytes() == once.vertices.tobytes() == unexploded[0].tobytes()
    assert np.array_equal(tensions.data[1], unexploded[1])
    assert np.array_equal(cells.data[0], cells.metadata["foambryo"]["unexploded_vertices"])


def test_a_later_inference_is_drawn_exploded(qtbot, viewer, docks):
    _, infer, results, _, view = docks
    _reconstructed_and_inferred(qtbot, viewer, docks)
    view.set_explode(0.25)
    infer.tension_box.setCurrentText("lami")
    _infer(qtbot, infer)
    run = RUNS[results.run_id]
    layer = next(lay for lay in viewer.layers if lay.name.endswith(" tensions"))
    assert layer.metadata["foambryo"]["explode"] == 0.25
    assert len(layer.data[0]) == len(interface_side_copy(run.mesh)[0].vertices)


def test_the_cut_is_one_world_plane_on_every_layer_and_the_labels(qtbot, viewer, docks):
    _, _, results, _, view = docks
    labels = _reconstructed_and_inferred(qtbot, viewer, docks, foam_anisotropic)
    display = view.shared.displays[results.run_id]
    view.axis_box.setCurrentIndex(2)  # normal to x
    view.keep_box.setCurrentIndex(1)  # keep the higher side
    view.position_slider.setValue(400)
    view.cut_box.setChecked(True)
    cut = view.cut()
    assert cut is not None and cut.axis == 2 and cut.keep == "higher"
    layers = [lay for lay in display.layers if lay in viewer.layers] + [labels]
    assert len(layers) >= 4  # cells, tensions, pressures, the Labels layer
    for layer in layers:
        (plane,) = layer.experimental_clipping_planes
        assert plane.enabled
        scale, translate = np.asarray(layer.scale)[-3:], np.asarray(layer.translate)[-3:]
        assert plane.position[2] * scale[2] + translate[2] == pytest.approx(cut.position, rel=1e-12)
        assert plane.normal[2] == 1.0 and plane.normal[0] == plane.normal[1] == 0.0
    assert tuple(labels.scale)[-1] != tuple(viewer.layers[f"{labels.name} cells"].scale)[-1]  # two frames, one plane
    assert "x =" in view.info.text()
    view.cut_box.setChecked(False)
    assert all(len(layer.experimental_clipping_planes) == 0 for layer in layers)


def test_tensions_only_has_no_pressure_anywhere(qtbot, viewer, docks, tmp_path):
    reconstruct, infer, results, export, _ = docks
    layer = _add(viewer, foam_cubic)
    _reconstruct(qtbot, reconstruct, layer.name)
    infer.pressures_box.setChecked(False)
    assert not infer.pressure_box.isEnabled() and not infer.base_pressure.isEnabled()
    _infer(qtbot, infer)
    assert "the tensions only" in infer.status.text()
    assert f"{layer.name} tensions" in viewer.layers
    assert f"{layer.name} pressures" not in viewer.layers
    assert results.cells.horizontalHeaderItem(0).text().startswith("the tensions only were inferred")
    assert "no pressures (the tensions only)" in results.summary.text()
    export.folder.setText(str(tmp_path / "out"))
    written = export.export()
    assert "tensions.csv" in written and "pressures.csv" not in written
    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["options"]["no_pressures"] is True
    assert "--no-pressures" in record["equivalent_commands"][1]


def test_an_earlier_pressure_layer_goes_when_the_pressures_are_off(qtbot, viewer, docks):
    _, infer, *_ = docks
    layer = _reconstructed_and_inferred(qtbot, viewer, docks)
    assert f"{layer.name} pressures" in viewer.layers
    infer.pressures_box.setChecked(False)
    _infer(qtbot, infer)
    assert f"{layer.name} pressures" not in viewer.layers


def test_the_diagnostics_tab_and_their_export(qtbot, viewer, docks, tmp_path):
    reconstruct, infer, results, export, _ = docks
    layer = _add(viewer, foam_cubic)
    _reconstruct(qtbot, reconstruct, layer.name)
    assert "No solve diagnostics" in results.diagnostics_note.text() or results.run_id is None
    infer.diagnostics_box.setCurrentText("spectrum")
    _infer(qtbot, infer)
    run = RUNS[results.run_id]
    assert results.diagnostics.rowCount() == len(run.outcome.records) > 0
    headers = [results.diagnostics.horizontalHeaderItem(c).text() for c in range(results.diagnostics.columnCount())]
    assert headers == list(tables.SOLVE_COLUMNS)
    assert results.diagnostics.item(0, 1).text() == run.outcome.records[0].system
    assert results.spectrum.pixmap() is not None and not results.spectrum.pixmap().isNull()
    export.folder.setText(str(tmp_path / "out"))
    written = export.export()
    assert {tables.SOLVE_TABLE, tables.SPECTRA} <= set(written)
    assert len(tables.read_solve_table(tmp_path / "out" / tables.SOLVE_TABLE)) == len(run.outcome.records)


def test_export_says_when_the_infer_docks_options_changed(qtbot, viewer, docks, tmp_path):
    _, infer, _, export, _ = docks
    _reconstructed_and_inferred(qtbot, viewer, docks)
    infer.diagnostics_box.setCurrentText("spectrum")  # after Infer: the inference exported has none
    export.folder.setText(str(tmp_path / "out"))
    written = export.export()
    assert tables.SPECTRA not in written
    assert "diagnostics differ from those of the inference exported: run Infer again" in export.status.text()


def _answer_the_dialog(button_text: str, seen: list) -> None:
    """Click `button_text` in the modal dialog once it is open (from inside its event loop)."""

    def click() -> None:
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(50, click)
            return
        seen.append(box.informativeText())
        for button in box.buttons():
            if button.text().replace("&", "") == button_text:
                button.click()
                return
        raise AssertionError(f"no {button_text} button in {[b.text() for b in box.buttons()]}")

    QTimer.singleShot(50, click)


def test_export_asks_before_replacing_and_replaces_only_foambryos_files(qtbot, viewer, docks, tmp_path):
    _, _, _, export, _ = docks
    _reconstructed_and_inferred(qtbot, viewer, docks)
    folder = tmp_path / "out"
    export.folder.setText(str(folder))
    first = export.export()
    assert first
    (folder / "notes.txt").write_text("mine")
    before = {name: (folder / name).stat().st_mtime_ns for name in first}
    seen: list = []
    _answer_the_dialog("Cancel", seen)
    assert export.export() == []
    assert "Cancelled" in export.status.text()
    assert all(name in seen[0] for name in first) and "notes.txt" not in seen[0]
    assert {name: (folder / name).stat().st_mtime_ns for name in first} == before
    _answer_the_dialog("Replace", seen)
    second = export.export()
    assert set(second) == set(first)
    assert "replacing" in export.status.text()
    assert (folder / "notes.txt").read_text() == "mine"
    assert all((folder / name).stat().st_mtime_ns >= before[name] for name in first)
    assert not hasattr(export, "overwrite")  # the check box is gone: the dialog asks


def test_the_background_paints_the_canvas_only(qtbot, viewer, docks, monkeypatch):
    from foambryo.colour_scales import BACKGROUND_COLOURS
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog

    _reconstructed_and_inferred(qtbot, viewer, docks)
    view = docks[-1]
    surfaces = [layer for layer in viewer.layers if hasattr(layer, "vertex_colors")]
    colours = [np.array(layer.vertex_colors, copy=True) for layer in surfaces]
    assert [view.background_box.itemText(i) for i in range(view.background_box.count())] == [
        "napari's theme",
        "black",
        "dark grey",
        "light grey",
        "white",
        "custom…",
    ]
    for index, name in enumerate(BACKGROUND_COLOURS, start=1):
        view.background_box.setCurrentIndex(index)
        view.background_box.activated.emit(index)
        assert np.allclose(viewer.canvas.background_color[:3], BACKGROUND_COLOURS[name], atol=1e-6)  # float32
    custom = view.background_box.count() - 1
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor.fromRgbF(0.1, 0.3, 0.5))
    view.background_box.activated.emit(custom)
    assert np.allclose(viewer.canvas.background_color[:3], (0.1, 0.3, 0.5), atol=1 / 255)
    assert view.background_box.currentIndex() == custom
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor())  # cancelled
    view.set_background("white")
    view.background_box.activated.emit(custom)
    assert np.allclose(viewer.canvas.background_color[:3], BACKGROUND_COLOURS["white"], atol=1e-6)
    assert view.background_box.currentText() == "white"
    view.set_background(None)
    assert viewer.canvas.background_color_override is None
    assert view.background_box.currentText() == "napari's theme"
    for layer, before in zip(surfaces, colours, strict=True):
        assert np.array_equal(layer.vertex_colors, before)
