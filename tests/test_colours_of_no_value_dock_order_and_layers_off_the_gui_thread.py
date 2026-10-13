"""The second round of the maintainer's feedback: colours, the docks, and the layers prepared off the GUI thread.

Driven as a user would, on a `ViewerModel` under Qt's offscreen platform:

* the tension colours are the library's new map (blue - light grey - red around the mean tension): what the
  library's functions give, nothing else (`test_display_copies_and_colours` pins the arrays);
* **what has no value** never shows a value's colour: an undetermined interface is not filled in the tensions layer,
  and its own layer draws it as a wireframe in the library's undetermined green over transparent faces; crop faces
  are glass (the library's fill at its opacity in the vertex colours, its edge colour in the wireframe); a selected
  row does not fill an undetermined interface;
* `min_distance` is on the Reconstruct dock itself, not in a fold; the docks are listed Reconstruct, Infer, View,
  Export, then Results;
* the layers' display copies and colours are **prepared in a worker thread** after an inference and when explode
  leaves 0, and napari receives them on the GUI thread; the arrays are those of the synchronous path, bitwise.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from foambryo.colour_scales import CROP_FACE_COLOR, CROP_FACE_EDGE_COLOR, CROP_FACE_OPACITY, UNDETERMINED_COLOR
from superqt import QCollapsible

from napari_foambryo import _layers
from napari_foambryo._display import interface_copy, undetermined_copy
from napari_foambryo._layers import RUNS, highlight, outline_layer_name, prepare_result_layers
from napari_foambryo._sample_data import foam_cropped, foam_cubic
from tests.test_view_dock_diagnostics_and_export_dialog import _reconstructed_and_inferred, docks, viewer  # noqa: F401
from tests.test_widgets_walk_through import TIMEOUT_MS


@pytest.fixture(autouse=True)
def _wait_for_background_workers(qtbot):
    yield
    from napari_foambryo._widgets import _LIVE_WORKERS

    qtbot.waitUntil(lambda: not _LIVE_WORKERS, timeout=TIMEOUT_MS)


def test_min_distance_is_on_the_reconstruct_dock_not_in_a_fold(docks):  # noqa: F811
    reconstruct = docks[0]
    assert not any(fold.isAncestorOf(reconstruct.min_distance) for fold in reconstruct.findChildren(QCollapsible))


def test_the_docks_are_listed_reconstruct_infer_view_export_results():
    import napari_foambryo

    manifest = yaml.safe_load((Path(napari_foambryo.__file__).parent / "napari.yaml").read_text())
    names = [widget["display_name"] for widget in manifest["contributions"]["widgets"]]
    assert names == ["Reconstruct", "Infer", "View", "Export", "Results"]


def test_an_undetermined_interface_is_an_outline_without_fill(qtbot, viewer, docks):  # noqa: F811
    _, _, results, _, _ = docks
    layer = _reconstructed_and_inferred(qtbot, viewer, docks)
    run = RUNS[results.run_id]
    tensions = dict(run.outcome.tensions)
    missing = next(iter(tensions))
    tensions[missing] = float("nan")
    run.outcome = type(run.outcome)(tensions, run.outcome.pressures, run.outcome.records, run.outcome.status,
                                    run.outcome.error, run.outcome.library_call)
    _layers.add_result_layers(viewer, run, layer.name)
    drawn = viewer.layers[f"{layer.name} tensions"]
    values = drawn.metadata["foambryo"]["values"]
    assert np.array_equal(drawn.vertex_colors[:, 3], np.where(np.isnan(values), 0.0, 1.0).astype(np.float32))
    outline = viewer.layers[outline_layer_name(drawn.name)]
    assert outline.wireframe.visible
    assert np.allclose(np.asarray(outline.wireframe.color)[:3], UNDETERMINED_COLOR)
    assert (outline.vertex_colors[:, 3] == 0).all()
    copy, _ = interface_copy(run.mesh)
    expected = undetermined_copy(copy, values)
    assert np.array_equal(outline.data[0], expected.vertices)
    keys = drawn.metadata["foambryo"]["interfaces"]
    highlight(viewer, drawn, keys.index(next(k for k in keys if k != missing)))  # select another interface
    assert (drawn.vertex_colors[np.isnan(values), 3] == 0).all()  # still unfilled
    run.outcome = type(run.outcome)(dict(run.outcome.tensions, **{}), run.outcome.pressures)
    run.outcome.tensions[missing] = 1.0
    _layers.add_result_layers(viewer, run, layer.name)
    assert outline_layer_name(drawn.name) not in viewer.layers  # no undetermined interface left: no outline layer


def test_crop_faces_are_edged_glass_and_never_a_pressure_colour(qtbot, viewer, docks):  # noqa: F811
    _, _, results, _, _ = docks
    layer = _reconstructed_and_inferred(qtbot, viewer, docks, sample=foam_cropped)
    crop = viewer.layers[f"{layer.name} crop faces"]
    assert np.allclose(crop.vertex_colors, np.asarray((*CROP_FACE_COLOR, CROP_FACE_OPACITY), dtype=np.float32))
    assert crop.wireframe.visible
    assert np.allclose(np.asarray(crop.wireframe.color)[:3], CROP_FACE_EDGE_COLOR)
    run = RUNS[results.run_id]
    pressures = viewer.layers[f"{layer.name} pressures"]
    assert not np.asarray(run.mesh.crop_faces)[pressures.metadata["foambryo"]["source_triangle"]].any()


def test_the_layers_are_prepared_off_the_gui_thread_and_are_the_synchronous_ones(qtbot, viewer, docks, monkeypatch):  # noqa: F811
    threads: list[bool] = []
    real = _layers.prepare_result_layers

    def recorded(*args: object, **kwargs: object):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(*args, **kwargs)

    import napari_foambryo._widgets as widgets

    monkeypatch.setattr(widgets, "prepare_result_layers", recorded)
    _, _, results, _, view = docks
    layer = _reconstructed_and_inferred(qtbot, viewer, docks, sample=foam_cubic)
    assert threads == [False]  # prepared in a worker thread
    run = RUNS[results.run_id]
    drawn = viewer.layers[f"{layer.name} tensions"]
    synchronous = prepare_result_layers(run, layer.name)[0]
    assert drawn.data[0].tobytes() == synchronous.vertices.tobytes()
    assert drawn.vertex_colors.tobytes() == synchronous.colours.tobytes()

    explode_threads: list[bool] = []
    real_explode = _layers.prepare_explode

    def explode_recorded(*args: object, **kwargs: object):
        explode_threads.append(threading.current_thread() is threading.main_thread())
        return real_explode(*args, **kwargs)

    monkeypatch.setattr(widgets, "prepare_explode", explode_recorded)
    view.explode_slider.setValue(40)
    qtbot.waitUntil(lambda: view.shared.displays[run.run_id].explode == 0.4, timeout=TIMEOUT_MS)
    assert explode_threads == [False]  # the copy change (0 to 0.4) prepared in a worker thread
    exploded = prepare_result_layers(run, layer.name, explode=0.4)[0]
    assert drawn.data[0].tobytes() == exploded.vertices.tobytes()


def test_a_one_mesh_inference_runs_in_a_child_process_that_cancel_stops(qtbot, viewer, docks, monkeypatch):  # noqa: F811
    """Every one-mesh Infer runs `infer_on_mesh` in a child process, whatever the solver, with Cancel enabled.

    In a worker thread it held Python's lock and froze napari 0.6-2 s on a 2 000-cell foam (the maintainer's choice of
    13 October: a child process, about a second of start-up on a small mesh).
    """
    import napari_foambryo._widgets as widgets

    targets: list[str] = []
    real = widgets.run_in_child

    def recorded(target: str, *args: object, **kwargs: object):
        targets.append(target)
        return real(target, *args, **kwargs)

    monkeypatch.setattr(widgets, "run_in_child", recorded)
    _, infer, *_ = docks
    _reconstructed_and_inferred(qtbot, viewer, docks, sample=foam_cubic)
    assert targets[-1] == "napari_foambryo._runs:infer_on_mesh"
    assert infer.solver_box.currentText() != "dense"  # the default solver too
    infer.start()
    assert infer.cancel_button.isEnabled()
    qtbot.waitUntil(lambda: infer._worker is None, timeout=TIMEOUT_MS)
