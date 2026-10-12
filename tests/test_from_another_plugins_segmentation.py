"""Gate G11 (tier T3): a Labels layer as another segmentation plugin leaves it gives the library's numbers.

The fixture imitates such a layer: `uint16`, labels not consecutive, the background labelled 1, no scale on the
labels (napari's default) but one on the image layer it was segmented from, a leading axis of length 1, held as a
dask array. Through the widgets (voxel size *from the image layer*, *medium is label 1*), the mesh and the tensions
and pressures are bitwise the library's on the equivalent clean mask: the same labels with the medium at 0, the
axis dropped, in memory, with the image layer's voxel size. The mask report is shown before Run; the run record
says where everything came from; Export includes the swapped mask, which the equivalent command needs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from foambryo import dcel_mesh_from_segmentation_mask, infer_forces

from napari_foambryo._layers import RUNS
from napari_foambryo._widgets import ExportWidget, InferWidget, ReconstructWidget, ResultsWidget
from tests.conftest import quietly, same_floats, sha

TIMEOUT_MS = 180_000


@pytest.fixture(autouse=True)
def _wait_for_background_workers(qtbot):
    """Let the widgets' background work (the mask report) finish before Qt is torn down."""
    yield
    from napari_foambryo._widgets import _LIVE_WORKERS

    qtbot.waitUntil(lambda: not _LIVE_WORKERS, timeout=TIMEOUT_MS)


@pytest.fixture
def another_plugins_output(samples):
    dask = pytest.importorskip("dask.array")
    foam, spacing = samples["anisotropic"]
    labels = np.where(foam == 0, 1, foam.astype(np.uint16) * 7 + 100).astype(np.uint16)  # background 1, sparse
    clean = np.where(labels == 1, 0, labels).astype(np.uint16)  # what the library should see
    return dask.from_array(labels[np.newaxis], chunks=(1, 16, 48, 48)), clean, spacing


def test_another_plugins_labels_give_the_librarys_numbers(qtbot, tmp_path, another_plugins_output):
    from napari.components import ViewerModel
    from napari.utils._proxies import PublicOnlyProxy

    lazy, clean, spacing = another_plugins_output
    viewer = ViewerModel()
    viewer.add_image(np.zeros(clean.shape, dtype=np.uint8), name="membranes", scale=spacing, units="um")
    viewer.add_labels(lazy, name="segmentation")
    reconstruct, infer, results, export = (
        cls(PublicOnlyProxy(viewer)) for cls in (ReconstructWidget, InferWidget, ResultsWidget, ExportWidget)
    )
    for widget in (reconstruct, infer, results, export):
        qtbot.addWidget(widget)
    reconstruct.select_layer("segmentation")
    assert "voxel size unknown" in reconstruct.spacing_label.text()  # the labels carry no scale
    reconstruct.spacing_from.setCurrentText("layer: membranes")
    assert "from layer 'membranes'" in reconstruct.spacing_label.text()
    reconstruct.medium_box.setValue(1)
    qtbot.waitUntil(
        lambda: "Mask report, for information (nothing is changed): 12 cells" in reconstruct.report_label.text(),
        timeout=TIMEOUT_MS,
    )
    assert reconstruct.run_button.isEnabled()
    reconstruct.start()
    qtbot.waitUntil(lambda: reconstruct._worker is None, timeout=TIMEOUT_MS)
    assert reconstruct.last_error is None, reconstruct.status.text()
    assert "segmentation (medium = 0)" in viewer.layers
    infer.start()
    qtbot.waitUntil(lambda: infer._worker is None, timeout=TIMEOUT_MS)

    run = RUNS[infer.session.current]
    library_mesh = quietly(dcel_mesh_from_segmentation_mask, clean, spacing=spacing)
    assert sha(run.mesh.v) == sha(library_mesh.v)
    assert sha(run.mesh.f) == sha(library_mesh.f)
    tensions, pressures = quietly(infer_forces, library_mesh)
    assert same_floats(run.outcome.tensions, tensions)
    assert same_floats(run.outcome.pressures, pressures)
    assert min(label for pair in tensions for label in pair) == 0
    assert all(label == 0 or label >= 107 for pair in tensions for label in pair)  # labels kept as they were

    source = run.source.as_record()
    assert source["spacing_from"].startswith("layer 'membranes'")
    assert source["medium_label"] == 1 and source["medium_swapped"]
    assert source["dropped_leading_axis"]
    assert any("lazy" in note for note in source["notes"])
    assert source["mask_report"]["cells"] == 12
    export.folder.setText(str(tmp_path / "exported"))
    written = export.export()
    assert "mask.npz" in written  # the swapped mask: the equivalent command needs it
    with np.load(tmp_path / "exported" / "mask.npz") as saved:
        assert np.array_equal(saved["mask"], clean)
        assert tuple(saved["spacing"]) == spacing
    record = json.loads((tmp_path / "exported" / "run.json").read_text())
    assert record["source_mask"]["medium_swapped"]


def test_editing_the_original_segmentation_marks_a_swapped_run_out_of_date(qtbot, samples):
    from napari.components import ViewerModel

    foam, spacing = samples["cubic"]
    viewer = ViewerModel()
    labels = viewer.add_labels(np.where(foam == 0, 1, foam + 1).astype(np.uint16), name="seg", scale=spacing)
    reconstruct, infer, export = (cls(viewer) for cls in (ReconstructWidget, InferWidget, ExportWidget))
    for widget in (reconstruct, infer, export):
        qtbot.addWidget(widget)
    reconstruct.select_layer("seg")
    reconstruct.medium_box.setValue(1)
    reconstruct.start()
    qtbot.waitUntil(lambda: reconstruct._worker is None, timeout=TIMEOUT_MS)
    infer.start()
    qtbot.waitUntil(lambda: infer._worker is None, timeout=TIMEOUT_MS)
    data = labels.data.copy()
    data[0, 0, 0] = 2
    labels.data = data  # the original, not the swapped copy
    assert RUNS[infer.session.current].out_of_date
    assert not export.export_button.isEnabled()


def test_the_mask_report_follows_the_labels(qtbot, samples):
    from napari.components import ViewerModel

    foam, spacing = samples["cubic"]
    viewer = ViewerModel()
    labels = viewer.add_labels(foam.copy(), name="seg", scale=spacing)
    reconstruct = ReconstructWidget(viewer)
    qtbot.addWidget(reconstruct)
    reconstruct.select_layer("seg")
    qtbot.waitUntil(lambda: "12 cells" in reconstruct.report_label.text(), timeout=TIMEOUT_MS)
    data = labels.data.copy()
    data[data == 3] = 0  # a cell removed, as a user's edit would
    labels.data = data
    qtbot.waitUntil(lambda: "11 cells" in reconstruct.report_label.text(), timeout=TIMEOUT_MS)
    labels.data[labels.data == 4] = 0  # in place, as painting does
    labels.events.paint(value=[])
    qtbot.waitUntil(lambda: "10 cells" in reconstruct.report_label.text(), timeout=TIMEOUT_MS)
