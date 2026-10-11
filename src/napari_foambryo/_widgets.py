"""The dock widgets: Reconstruct, Infer, Results and Export.

Every option's default is read from the library function's signature when the widget is built, as the command
does; advanced options are folded away. Long steps run off the GUI thread: a reconstruction in a child process that
Cancel stops (`_runner`), an inference in a worker thread. Every warning the library raises is listed once, with
its count, in the Results dock.
"""

from __future__ import annotations

import inspect
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from foambryo import infer_forces
from foambryo import results as tables
from foambryo.colour_scales import save_colour_bars
from foambryo.segmentation_masks import SegmentationMaskError
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from napari_foambryo._estimate import estimate_reconstruction, memory_verdict
from napari_foambryo._layers import (
    RUNS,
    add_ensemble_layer,
    add_mesh_layers,
    add_result_layers,
    highlight,
    new_run_id,
)
from napari_foambryo._masks import array_sha256, prepare_mask, swap_medium_label
from napari_foambryo._runner import ChildCancelledError, ChildFailedError, run_in_child
from napari_foambryo._runs import EnsembleOptions, InferenceOptions, Run, SourceMask, export_run, infer_on_mesh
from napari_foambryo._session import RunDisplay, session_for
from napari_foambryo._spacing import THIS_LAYER, TYPED, resolve_spacing

__all__ = ["ExportWidget", "InferWidget", "ReconstructWidget", "ResultsWidget"]

AMBER, RED = "#b26b00", "#c0392b"


def _signature_default(function: Any, parameter: str) -> Any:  # noqa: ANN401
    return inspect.signature(function).parameters[parameter].default


def _labels_layers(viewer: Any) -> list:  # noqa: ANN401
    from napari.layers import Labels

    return [layer for layer in viewer.layers if isinstance(layer, Labels)]


#: Workers still running, held until they finish: a dock closed meanwhile must not let one be collected half-way.
_LIVE_WORKERS: set = set()


def _start_worker(function: Any, *args: Any, on_returned: Any, on_errored: Any) -> Any:  # noqa: ANN401
    from napari.qt.threading import create_worker

    worker = create_worker(function, *args, _start_thread=False)
    worker.returned.connect(_unless_closed(on_returned))
    worker.errored.connect(_unless_closed(on_errored))
    _LIVE_WORKERS.add(worker)
    worker.finished.connect(lambda: _LIVE_WORKERS.discard(worker))
    worker.start()
    return worker


def _unless_closed(handler: Any) -> Any:  # noqa: ANN401
    """A worker's handler that does nothing once its dock was closed (its Qt widgets deleted) during the run."""

    def handle(value: Any) -> None:  # noqa: ANN401
        try:
            handler(value)
        except RuntimeError as error:
            if "has been deleted" not in str(error):
                raise

    return handle


def _status(label: QLabel, text: str, colour: str | None = None) -> None:
    label.setText(text)
    label.setStyleSheet(f"color: {colour};" if colour else "")


def _folded(title: str, content: QWidget) -> QCollapsible:
    collapsible = QCollapsible(title)
    collapsible.addWidget(content)
    collapsible.collapse(animate=False)
    return collapsible


# ---------------------------------------------------------------------------------------------
# Reconstruct
# ---------------------------------------------------------------------------------------------
class ReconstructWidget(QWidget):
    """A Labels layer, its voxel size, the estimate, Run and Cancel."""

    def __init__(self, napari_viewer: Any) -> None:  # noqa: ANN401
        """Build the form; defaults are the library's."""
        from dw3d import get_default_mesh_reconstruction_algorithm

        super().__init__()
        self.viewer = napari_viewer
        self.session = session_for(napari_viewer)
        self._cancel = threading.Event()
        self._memory = {"rss": 0, "started": 0.0}
        self._worker = None
        self.last_error: str | None = None

        self.layer_box = QComboBox()
        self.spacing_from = QComboBox()
        self.spacing_boxes = [QDoubleSpinBox() for _ in range(3)]
        for box in self.spacing_boxes:
            box.setDecimals(6)
            box.setRange(1e-6, 1e6)
            box.setValue(1.0)
        self.unit_edit = QLineEdit("um")
        self.confirm_voxels = QCheckBox("the voxel size is unknown: use voxel units (1 × 1 × 1)")
        self.medium_box = QSpinBox()
        self.medium_box.setRange(0, 2**31 - 1)
        self.medium_box.setValue(0)
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 1000)
        self.min_distance.setValue(int(_signature_default(get_default_mesh_reconstruction_algorithm, "min_distance")))
        self.spacing_label = QLabel()
        self.spacing_label.setWordWrap(True)
        self.estimate_label = QLabel()
        self.report_label = QLabel()
        self.report_label.setWordWrap(True)
        self._reports: dict[tuple, Any] = {}
        self._report_worker = None
        self.estimate_label.setWordWrap(True)
        self.run_anyway = QCheckBox("run anyway")
        self.run_button = QPushButton("Reconstruct")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.status = QLabel()
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Labels layer", self.layer_box)
        form.addRow("Voxel size from", self.spacing_from)
        spacing_row = QHBoxLayout()
        for box in self.spacing_boxes:
            spacing_row.addWidget(box)
        form.addRow("Voxel size (z, y, x)", spacing_row)
        form.addRow("Unit", self.unit_edit)
        form.addRow(self.confirm_voxels)
        form.addRow("Medium is label", self.medium_box)
        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        advanced_form.addRow("min_distance (cubic voxels)", self.min_distance)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(_folded("Advanced", advanced))
        layout.addWidget(self.spacing_label)
        layout.addWidget(self.estimate_label)
        layout.addWidget(self.report_label)
        layout.addWidget(self.run_anyway)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._show_progress)
        self.layer_box.currentIndexChanged.connect(self.refresh)
        self.medium_box.valueChanged.connect(self.refresh)
        self.spacing_from.currentIndexChanged.connect(self.refresh)
        for box in self.spacing_boxes:
            box.valueChanged.connect(self.refresh)
        self.unit_edit.textChanged.connect(self.refresh)
        self.confirm_voxels.toggled.connect(self.refresh)
        self.run_anyway.toggled.connect(self.refresh)
        self.min_distance.valueChanged.connect(self.refresh)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self._cancel.set)
        napari_viewer.layers.events.inserted.connect(self._refill)
        napari_viewer.layers.events.removed.connect(self._refill)
        self._refill()

    # -- the form --------------------------------------------------------------------------------
    def _refill(self, _event: Any = None) -> None:  # noqa: ANN401
        current = self.layer_box.currentText()
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        for layer in _labels_layers(self.viewer):
            self.layer_box.addItem(layer.name)
        if current and self.layer_box.findText(current) >= 0:
            self.layer_box.setCurrentText(current)
        self.layer_box.blockSignals(False)
        chosen = self.spacing_from.currentText()
        self.spacing_from.blockSignals(True)
        self.spacing_from.clear()
        self.spacing_from.addItem(THIS_LAYER)
        for layer in self.viewer.layers:
            if layer.name != self.layer_box.currentText() and layer.ndim >= 3:
                self.spacing_from.addItem(f"layer: {layer.name}")
        self.spacing_from.addItem(TYPED)
        if chosen and self.spacing_from.findText(chosen) >= 0:
            self.spacing_from.setCurrentText(chosen)
        self.spacing_from.blockSignals(False)
        self.refresh()

    def select_layer(self, name: str) -> None:
        """Choose the Labels layer `name` (the Layers menu's *Reconstruct with foambryo* does this)."""
        self._refill()
        index = self.layer_box.findText(name)
        if index >= 0:
            self.layer_box.setCurrentIndex(index)

    def labels_layer(self) -> Any:  # noqa: ANN401
        """The chosen Labels layer, or `None`."""
        name = self.layer_box.currentText()
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _choice(self) -> Any:  # noqa: ANN401
        labels = self.labels_layer()
        if labels is None:
            return None
        mode = self.spacing_from.currentText() or THIS_LAYER
        other = None
        if mode.startswith("layer: "):
            name = mode.removeprefix("layer: ")
            other = self.viewer.layers[name] if name in self.viewer.layers else None  # noqa: SIM401 - LayerList has no get
            mode = "layer"
        shape = labels.data[0].shape[-3:] if labels.multiscale else labels.data.shape[-3:]
        return resolve_spacing(
            labels,
            mode,
            mask_shape=tuple(shape),
            other=other,
            typed=tuple(box.value() for box in self.spacing_boxes),
            typed_unit=self.unit_edit.text().strip(),
            voxel_units_confirmed=self.confirm_voxels.isChecked(),
        )

    def refresh(self, *_args: Any) -> None:  # noqa: ANN401
        """Recompute the spacing message, the estimate and whether Run is possible."""
        typed = self.spacing_from.currentText() == TYPED
        for box in self.spacing_boxes:
            box.setEnabled(typed)
        self.unit_edit.setEnabled(typed)
        choice = self._choice()
        busy = self._worker is not None
        if choice is None:
            _status(self.spacing_label, "add or open a Labels layer (a segmentation)")
            self.estimate_label.clear()
            self.confirm_voxels.setVisible(False)
            self.run_anyway.setVisible(False)
            self.run_button.setEnabled(False)
            return
        self.confirm_voxels.setVisible(choice.needs_confirmation or self.confirm_voxels.isChecked())
        self._request_report()
        if not typed:
            for box, value in zip(self.spacing_boxes, choice.spacing, strict=True):
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
            self.unit_edit.blockSignals(True)
            self.unit_edit.setText(choice.unit)
            self.unit_edit.blockSignals(False)
        if choice.problem:
            _status(self.spacing_label, choice.problem, AMBER if choice.needs_confirmation else RED)
            self.estimate_label.clear()
            self.run_anyway.setVisible(False)
            self.run_button.setEnabled(False)
            return
        _status(self.spacing_label, choice.message)
        labels = self.labels_layer()
        shape = labels.data[0].shape[-3:] if labels.multiscale else labels.data.shape[-3:]
        estimate = estimate_reconstruction(tuple(shape), choice.spacing)
        import psutil

        memory = psutil.virtual_memory()
        verdict = memory_verdict(estimate, memory.total, memory.available)
        _status(
            self.estimate_label,
            f"Estimate: {estimate.text()} ({estimate.voxels:.3g} voxels after resampling; "
            f"the band is in-sample, not a guarantee). {verdict.message}",
            {"ok": None, "amber": AMBER, "red": RED}[verdict.level],
        )
        self.run_anyway.setVisible(verdict.needs_run_anyway)
        self.run_button.setEnabled(not busy and (not verdict.needs_run_anyway or self.run_anyway.isChecked()))

    # -- the run -----------------------------------------------------------------------------------
    def start(self) -> None:
        """Prepare the mask in this thread, reconstruct it in a child process."""
        labels = self.labels_layer()
        choice = self._choice()
        if labels is None or choice is None or choice.problem:
            return
        try:
            prepared = prepare_mask(labels.data, multiscale=labels.multiscale, name=f"'{labels.name}'")
            medium = self.medium_box.value()
            mask = swap_medium_label(prepared.mask, medium)
        except SegmentationMaskError as error:
            _status(self.status, str(error), RED)
            return
        source = SourceMask(
            layer_name=labels.name,
            shape=tuple(prepared.mask.shape),
            dtype=str(prepared.mask.dtype),
            sha256=array_sha256(prepared.mask),
            spacing=choice.spacing,
            spacing_unit=choice.unit,
            spacing_from=choice.spacing_from,
            medium_label=medium,
            notes=prepared.notes,
            dropped_leading_axis=prepared.dropped_leading_axis,
            multiscale_level=prepared.multiscale_level,
            origin=_layer_origin(labels),
            mask_report=self._report_record(labels),
        )
        base_name = labels.name
        if medium != 0:
            base_name = f"{labels.name} (medium = 0)"
            if base_name in self.viewer.layers:
                self.viewer.layers.remove(base_name)
            swapped = self.viewer.add_labels(
                mask, name=base_name, scale=tuple(labels.scale[-3:]), translate=tuple(labels.translate[-3:])
            )
            labels_for_display = swapped
        else:
            labels_for_display = labels
        self._cancel.clear()
        self._memory.update(rss=0, started=time.perf_counter())
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        notes = " ".join(prepared.notes)
        _status(self.status, f"Reconstructing in a child process… {notes}".strip())
        self._timer.start()
        kwargs = {
            "segmentation_mask": mask,
            "spacing": choice.spacing,
            "spacing_unit": choice.unit,
            "min_distance": self.min_distance.value(),
        }

        def reconstruct() -> Any:  # noqa: ANN401
            return run_in_child(
                "foambryo.io:reconstruct_mask",
                kwargs,
                should_cancel=self._cancel.is_set,
                on_memory=lambda rss: self._memory.update(rss=rss),
            )

        def returned(result: Any) -> None:  # noqa: ANN401
            reconstruction = result.value
            run = Run(
                run_id=new_run_id(),
                mesh=reconstruction.mesh(),
                source=source,
                mask=mask,
                reconstruction=reconstruction,
                rec_bytes=reconstruction.rec_bytes,
                reconstruct_warnings=result.warnings_summary,
                reconstruct_seconds=result.seconds,
                peak_rss_bytes=result.peak_rss_bytes,
            )
            RUNS[run.run_id] = run
            display = RunDisplay(base_name, labels_for_display, choice.spacing)
            self.session.add(run, display)
            for name in add_mesh_layers(self.viewer, run, base_name, labels_for_display, choice.spacing):
                display.layers.append(self.viewer.layers[name])
            hidden = ""
            if self.viewer.dims.ndisplay == 3 and labels_for_display.visible:
                labels_for_display.visible = False  # in 3-D the labels' voxels hide the mesh
                hidden = f" The labels layer '{labels_for_display.name}' is hidden (its voxels hide the mesh in 3-D)."
            self._finish(
                f"{reconstruction.summary()} in {result.seconds:.1f} s, peak "
                f"{result.peak_rss_bytes / 2**30:.2f} GiB.{hidden} Next: Infer."
            )

        def errored(error: BaseException) -> None:
            if isinstance(error, ChildCancelledError):
                self._finish("Cancelled: nothing was added.", AMBER)
            elif isinstance(error, ChildFailedError):
                self._finish(f"The reconstruction refused the mask: {error.message}", RED)
            else:
                self._finish(f"Failed: {type(error).__name__}: {error}", RED)

        self._worker = _start_worker(reconstruct, on_returned=returned, on_errored=errored)

    # -- the mask report ---------------------------------------------------------------------------
    def _report_key(self, labels: Any) -> tuple:  # noqa: ANN401
        data = labels.data[0] if labels.multiscale else labels.data
        return (labels.name, id(data), self.medium_box.value())

    def _report_record(self, labels: Any) -> dict | None:  # noqa: ANN401
        report = self._reports.get(self._report_key(labels))
        return None if report is None else report.as_record()

    def _request_report(self) -> None:
        """Describe the chosen mask (foambryo's `segmentation_mask_report`) off the GUI thread; nothing is changed."""
        from foambryo.segmentation_masks import segmentation_mask_report

        labels = self.labels_layer()
        if labels is None:
            self.report_label.clear()
            return
        key = self._report_key(labels)
        if key in self._reports:
            self._show_report(key)
            return
        if self._report_worker is not None:
            return
        _status(self.report_label, "Mask report: counting…")
        data, multiscale, name, medium = labels.data, labels.multiscale, f"'{labels.name}'", self.medium_box.value()

        def describe() -> Any:  # noqa: ANN401
            prepared = prepare_mask(data, multiscale=multiscale, name=name)
            return segmentation_mask_report(swap_medium_label(prepared.mask, medium))

        def returned(report: Any) -> None:  # noqa: ANN401
            self._report_worker = None
            self._reports[key] = report
            self._request_report()

        def errored(error: BaseException) -> None:
            self._report_worker = None
            _status(self.report_label, f"Mask report: {error}", RED)

        self._report_worker = _start_worker(describe, on_returned=returned, on_errored=errored)

    def _show_report(self, key: tuple) -> None:
        report = self._reports[key]
        notable = report.cells_in_several_pieces or report.enclosed_medium_pieces or report.small_labels
        text = f"Mask report (nothing is changed): {report.summary().removeprefix('mask: ')}."
        if report.cells_in_several_pieces:
            shown = ", ".join(f"{k} ({v})" for k, v in list(report.cells_in_several_pieces.items())[:5])
            text += f" In several pieces: {shown}{'…' if len(report.cells_in_several_pieces) > 5 else ''}."
        if report.small_labels:
            text += f" Very small: {', '.join(str(k) for k in list(report.small_labels)[:8])}."
        _status(self.report_label, text, AMBER if notable else None)

    def _show_progress(self) -> None:
        elapsed = time.perf_counter() - self._memory["started"]
        _status(
            self.status,
            f"Reconstructing in a child process: {elapsed:.0f} s, "
            f"{self._memory['rss'] / 2**30:.2f} GiB now. Cancel stops it.",
        )

    def _finish(self, message: str, colour: str | None = None) -> None:
        self._timer.stop()
        self._worker = None
        self.cancel_button.setEnabled(False)
        self.last_error = message if colour == RED else None
        _status(self.status, message, colour)
        self.refresh()


def _layer_origin(layer: Any) -> dict[str, Any]:  # noqa: ANN401
    source = getattr(layer, "source", None)
    if source is None:
        return {}
    origin = {}
    for key in ("path", "reader_plugin", "sample", "widget", "parent"):
        value = getattr(source, key, None)
        if value is not None:
            origin[key] = str(value)
    return origin


# ---------------------------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------------------------
class InferWidget(QWidget):
    """One mesh (the tension and pressure methods) or a seed ensemble; Run, Cancel."""

    ONE_MESH = "one mesh"
    ENSEMBLE = "seed ensemble"

    def __init__(self, napari_viewer: Any) -> None:  # noqa: ANN401
        """Build the form; defaults are `infer_forces`'s and `infer_tensions_over_seeds`'s."""
        from foambryo import DEFAULT_ENSEMBLE_SEEDS, infer_tensions_over_seeds
        from foambryo.second_order_angle_reading import ANGLE_READINGS
        from foambryo.seed_ensembles import COMBINE_OPTIONS

        super().__init__()
        self.viewer = napari_viewer
        self.session = session_for(napari_viewer)
        self._worker = None
        self._cancel = threading.Event()
        self._progress = {"text": "", "started": 0.0, "rss": 0}
        self.mesh_box = QComboBox()
        self.mesh_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mode_box = QComboBox()
        self.mode_box.addItems([self.ONE_MESH, self.ENSEMBLE])
        self.tension_box = QComboBox()
        self.tension_box.addItems(list(tables.TENSION_METHODS))
        self.tension_box.setCurrentText(tables.option_name(_signature_default(infer_forces, "mode_tension")))
        self.pressure_box = QComboBox()
        self.pressure_box.addItems(list(tables.PRESSURE_METHODS))
        self.pressure_box.setCurrentText(tables.option_name(_signature_default(infer_forces, "mode_pressure")))
        self.mean_tension = QLineEdit(repr(float(_signature_default(infer_forces, "mean_tension"))))
        self.base_pressure = QLineEdit(repr(float(_signature_default(infer_forces, "base_pressure"))))
        self.solver_box = QComboBox()
        self.solver_box.addItems(["auto", "dense", "sparse"])
        self.solver_box.setCurrentText(_signature_default(infer_forces, "solver"))
        self.diagnostics_box = QComboBox()
        self.diagnostics_box.addItems(list(tables.DIAGNOSTICS))
        library = _signature_default(infer_forces, "diagnostics")
        self.diagnostics_box.setCurrentText(next(k for k, v in tables.DIAGNOSTICS.items() if v is library))
        self.seeds_box = QSpinBox()
        self.seeds_box.setRange(2, 64)
        self.seeds_box.setValue(len(DEFAULT_ENSEMBLE_SEEDS))
        self.combine_box = QComboBox()
        self.combine_box.addItems(list(COMBINE_OPTIONS))
        self.combine_box.setCurrentText(_signature_default(infer_tensions_over_seeds, "combine"))
        self.reading_box = QComboBox()
        self.reading_box.addItems(list(ANGLE_READINGS))
        self.reading_box.setCurrentText(_signature_default(infer_tensions_over_seeds, "angle_reading"))
        self.workers_box = QSpinBox()
        self.workers_box.setRange(1, 64)
        self.workers_box.setValue(int(_signature_default(infer_tensions_over_seeds, "max_workers")))
        self.ensemble_estimate = QLabel()
        self.ensemble_estimate.setWordWrap(True)
        self.run_button = QPushButton("Infer")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.mesh_info = QLabel()
        self.mesh_info.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Mesh", self.mesh_box)
        form.addRow("Mode", self.mode_box)
        form.addRow("Tension method", self.tension_box)
        self.pressure_label = QLabel("Pressure method")
        form.addRow(self.pressure_label, self.pressure_box)
        self.ensemble_form = QWidget()
        ensemble_form = QFormLayout(self.ensemble_form)
        ensemble_form.setContentsMargins(0, 0, 0, 0)
        ensemble_form.addRow("Seeds (0 … N−1), N", self.seeds_box)
        ensemble_form.addRow("Workers (processes)", self.workers_box)
        ensemble_form.addRow(self.ensemble_estimate)
        self.one_mesh_advanced = QWidget()
        advanced_form = QFormLayout(self.one_mesh_advanced)
        advanced_form.addRow("Mean tension", self.mean_tension)
        advanced_form.addRow("Base pressure (medium)", self.base_pressure)
        advanced_form.addRow("Solver", self.solver_box)
        advanced_form.addRow("Diagnostics", self.diagnostics_box)
        self.ensemble_advanced = QWidget()
        ensemble_advanced_form = QFormLayout(self.ensemble_advanced)
        ensemble_advanced_form.addRow("Combine", self.combine_box)
        ensemble_advanced_form.addRow("Angle reading", self.reading_box)
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.addWidget(self.one_mesh_advanced)
        advanced_layout.addWidget(self.ensemble_advanced)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.ensemble_form)
        layout.addWidget(_folded("Advanced", advanced))
        layout.addWidget(self.mesh_info)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._show_progress)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self._cancel.set)
        self.mesh_box.currentIndexChanged.connect(self._mesh_changed)
        self.mode_box.currentIndexChanged.connect(self._mesh_changed)
        self.seeds_box.valueChanged.connect(self._mesh_changed)
        self.workers_box.valueChanged.connect(self._mesh_changed)
        self.session.run_added.connect(self._refill)
        self.session.out_of_date.connect(self._refill)
        napari_viewer.layers.events.inserted.connect(self._refill)
        napari_viewer.layers.events.removed.connect(self._refill)
        self._refill()

    def _refill(self, *_args: Any) -> None:  # noqa: ANN401
        from napari_foambryo._layers import run_of_layer

        for layer in self.viewer.layers:  # meshes opened with the reader join the session
            run = run_of_layer(layer)
            if run is not None and run.run_id not in self.session.displays:
                base = layer.metadata["foambryo"].get("base_name", layer.name)
                display = RunDisplay(base)
                self.session.displays[run.run_id] = display
                self.session.order.append(run.run_id)
                self.session.current = run.run_id
            if run is not None:
                shown = self.session.displays[run.run_id]
                if layer not in shown.layers:
                    shown.layers.append(layer)
        self.mesh_box.blockSignals(True)
        self.mesh_box.clear()
        for run_id in self.session.order:
            run = RUNS[run_id]
            suffix = " [out of date]" if run.out_of_date else ""
            self.mesh_box.addItem(f"{self.session.displays[run_id].base_name}{suffix}", run_id)
        if self.session.current in self.session.order:
            self.mesh_box.setCurrentIndex(self.session.order.index(self.session.current))
        self.mesh_box.blockSignals(False)
        self._mesh_changed()

    def ensemble_mode(self) -> bool:
        """Whether the form is in seed-ensemble mode."""
        return self.mode_box.currentText() == self.ENSEMBLE

    def _mesh_changed(self, *_args: Any) -> None:  # noqa: ANN401
        run = RUNS.get(self.mesh_box.currentData() or "")
        ensemble = self.ensemble_mode()
        self.pressure_label.setVisible(not ensemble)
        self.pressure_box.setVisible(not ensemble)
        self.ensemble_form.setVisible(ensemble)
        self.one_mesh_advanced.setVisible(not ensemble)
        self.ensemble_advanced.setVisible(ensemble)
        cropped = run is not None and bool(np.any(run.mesh.crop_faces))
        model = self.pressure_box.model()
        for index, member in enumerate(tables.PRESSURE_METHODS.values()):
            allowed = not cropped or member.has_crop_face_formulation
            model.item(index).setEnabled(allowed)
            if not allowed and self.pressure_box.currentIndex() == index:
                self.pressure_box.setCurrentText(tables.option_name(_signature_default(infer_forces, "mode_pressure")))
        possible = run is not None and (not ensemble or run.mask is not None)
        self.run_button.setEnabled(possible and self._worker is None)
        if run is None:
            _status(self.mesh_info, "Reconstruct a mask, or open a .rec file, first.")
        elif ensemble and run.mask is None:
            _status(
                self.mesh_info,
                "A seed ensemble reconstructs the mask again at each seed: it needs a mesh "
                "reconstructed here, not one opened from a file.",
                AMBER,
            )
        elif cropped and not ensemble:
            _status(
                self.mesh_info,
                f"This mesh has crop faces ({len(run.mesh.cut_cells)} cut cells): only pressure "
                "methods with a formulation for crop faces are offered; cut cells' pressures are "
                "nan.",
            )
        else:
            self.mesh_info.clear()
        if ensemble and run is not None and run.source is not None:
            self._show_ensemble_estimate(run)

    def _show_ensemble_estimate(self, run: Run) -> None:
        import math

        import psutil

        per_seed = estimate_reconstruction(tuple(run.source.shape), run.source.spacing)
        seeds, workers = self.seeds_box.value(), self.workers_box.value()
        rounds = math.ceil(seeds / workers)
        memory = psutil.virtual_memory()
        suggested = max(1, int(memory.available / 2**30 // max(per_seed.peak_gib[2], 1e-9)))
        text = (
            f"Estimate: {rounds} round(s) of {workers} seed(s): about {rounds * per_seed.seconds[1] / 60:.1f} min "
            f"({rounds * per_seed.seconds[0] / 60:.1f}–{rounds * per_seed.seconds[2] / 60:.1f}) and "
            f"{workers * per_seed.peak_gib[1]:.1f} GiB ({workers * per_seed.peak_gib[2]:.1f} at most); "
            f"{memory.available / 2**30:.0f} GiB free allows about {suggested} worker(s). "
            "The seed spread is a lower bound on the error, not an error bar."
        )
        _status(self.ensemble_estimate, text, AMBER if workers > suggested else None)

    def options(self) -> InferenceOptions:
        """The one-mesh options."""
        return InferenceOptions(
            tension_method=self.tension_box.currentText(),
            pressure_method=self.pressure_box.currentText(),
            mean_tension=float(self.mean_tension.text()),
            base_pressure=float(self.base_pressure.text()),
            solver=self.solver_box.currentText(),
            diagnostics=self.diagnostics_box.currentText(),
        )

    def ensemble_options(self) -> EnsembleOptions:
        """The seed-ensemble options."""
        return EnsembleOptions(
            seeds=self.seeds_box.value(),
            tension_method=self.tension_box.currentText(),
            combine=self.combine_box.currentText(),
            angle_reading=self.reading_box.currentText(),
            max_workers=self.workers_box.value(),
        )

    def start(self) -> None:
        """Infer on the chosen mesh: one mesh in a worker thread (a child process with the dense solver), or a seed
        ensemble in a child process."""
        run = RUNS.get(self.mesh_box.currentData() or "")
        if run is None:
            return
        self.session.current = run.run_id
        if self.ensemble_mode():
            self._start_ensemble(run)
            return
        try:
            options = self.options()
        except ValueError as error:
            _status(self.status, f"mean tension and base pressure must be numbers: {error}", RED)
            return
        started = datetime.now(UTC).isoformat(timespec="seconds")

        def returned(result: Any) -> None:  # noqa: ANN401
            outcome, warnings_summary, seconds = result.value if hasattr(result, "value") else result
            run.options, run.outcome, run.infer_warnings = options, outcome, warnings_summary
            run.infer_seconds, run.infer_date, run.last_inference = seconds, started, "one mesh"
            display = self.session.displays[run.run_id]
            for name in add_result_layers(
                self.viewer, run, display.base_name, display.source_layer, display.spacing_used
            ):
                layer = self.viewer.layers[name]
                if layer not in display.layers:
                    display.layers.append(layer)
            shown = _show_one_result(display.layers)
            colour = None if outcome.status == "complete" else (AMBER if outcome.tensions is not None else RED)
            text = f"{outcome.status} in {seconds:.1f} s"
            if outcome.error:
                text += f": {outcome.error}"
            if shown:
                text += f". Shown: {shown}; the other result layers are hidden (tick their eye to show them)."
            self._finish(text, colour)
            self.session.results_changed.emit(run.run_id)

        if options.solver == "dense":  # the one solve measured in minutes on a large tissue: cancellable
            self._begin("Inferring with the dense solver in a child process…")
            self._worker = _start_worker(
                self._in_child,
                "napari_foambryo._runs:infer_on_mesh",
                {"mesh": run.mesh, "options": options},
                False,
                on_returned=returned,
                on_errored=self._errored,
            )
        else:
            self._begin("Inferring the tensions, then the pressures…", cancellable=False)
            self._worker = _start_worker(
                infer_on_mesh, run.mesh, options, on_returned=returned, on_errored=self._errored
            )

    def _start_ensemble(self, run: Run) -> None:
        options = self.ensemble_options()
        started = datetime.now(UTC).isoformat(timespec="seconds")
        kwargs = options.library_kwargs(run.mask, run.source.spacing, run.reconstruction.min_distance)

        def returned(result: Any) -> None:  # noqa: ANN401
            run.ensemble, run.ensemble_options = result.value, options
            run.ensemble_warnings, run.ensemble_seconds = result.warnings_summary, result.seconds
            run.ensemble_date, run.ensemble_peak_rss_bytes = started, result.peak_rss_bytes
            run.last_inference = "ensemble"
            display = self.session.displays[run.run_id]
            name = add_ensemble_layer(self.viewer, run, display.base_name, display.source_layer, display.spacing_used)
            layer = self.viewer.layers[name]
            if layer not in display.layers:
                display.layers.append(layer)
            for other in display.layers:
                if other.metadata.get("foambryo", {}).get("kind") in ("cells", "tensions", "pressures"):
                    other.visible = False
            flagged = sum(result.value.found_by_fewer_than_half_the_seeds.values())
            self._finish(
                f"seed ensemble of {options.seeds} seeds in {result.seconds:.1f} s, peak "
                f"{result.peak_rss_bytes / 2**30:.2f} GiB: {len(result.value.tensions)} interfaces, "
                f"{flagged} found by fewer than half of the seeds. Shown: {name}."
            )
            self.session.results_changed.emit(run.run_id)

        self._begin(f"Seed ensemble: 0 of {options.seeds} seeds done…")
        self._worker = _start_worker(
            self._in_child,
            "foambryo.seed_ensembles:infer_tensions_over_seeds",
            kwargs,
            True,
            on_returned=returned,
            on_errored=self._errored,
        )

    def _in_child(self, target: str, kwargs: dict, with_progress: bool) -> Any:  # noqa: ANN401
        def progress(done: int, total: int, seed: int) -> None:
            self._progress["text"] = f"Seed ensemble: {done} of {total} seeds done (last: seed {seed})"

        return run_in_child(
            target,
            kwargs,
            should_cancel=self._cancel.is_set,
            on_progress=progress if with_progress else None,
            on_memory=lambda rss: self._progress.update(rss=rss),
        )

    def _begin(self, text: str, *, cancellable: bool = True) -> None:
        self._cancel.clear()
        self._progress.update(text=text, started=time.perf_counter(), rss=0)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(cancellable)
        _status(self.status, text)
        if cancellable:
            self._timer.start()

    def _show_progress(self) -> None:
        elapsed = time.perf_counter() - self._progress["started"]
        _status(
            self.status,
            f"{self._progress['text']} — {elapsed:.0f} s, {self._progress['rss'] / 2**30:.2f} GiB "
            "now. Cancel stops it.",
        )

    def _finish(self, text: str, colour: str | None = None) -> None:
        self._timer.stop()
        self._worker = None
        self.cancel_button.setEnabled(False)
        _status(self.status, text, colour)
        self._mesh_changed()

    def _errored(self, error: BaseException) -> None:
        if isinstance(error, ChildCancelledError):
            self._finish("Cancelled: nothing was changed.", AMBER)
        elif isinstance(error, ChildFailedError):
            self._finish(f"The library refused: {error.message}", RED)
        else:
            self._finish(f"Failed: {type(error).__name__}: {error}", RED)


def _show_one_result(layers: list) -> str:
    """Show the tensions layer and hide the cells and pressures (they share the geometry); returns what is shown."""
    kinds = {layer.metadata.get("foambryo", {}).get("kind"): layer for layer in layers}
    if "tensions" not in kinds:
        return ""
    for kind, layer in kinds.items():
        if kind in ("cells", "pressures", "tensions"):
            layer.visible = kind == "tensions"
    return kinds["tensions"].name


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------
class ResultsWidget(QWidget):
    """The interfaces and cells tables, the colour bars, the warnings; a row selects its interface or cell."""

    def __init__(self, napari_viewer: Any) -> None:  # noqa: ANN401
        """Empty until an inference."""
        super().__init__()
        self.viewer = napari_viewer
        self.session = session_for(napari_viewer)
        self.run_id: str | None = None
        self.tabs = QTabWidget()
        self.interfaces = self._table()
        self.cells = self._table()
        self.tabs.addTab(self.interfaces, "Interfaces")
        self.tabs.addTab(self.cells, "Cells")
        self.flagged_only = QCheckBox("flagged only (undetermined, cut)")
        self.bars = QLabel()
        self.bars.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMaximumHeight(110)
        self.csv_button = QPushButton("Export CSV of this table…")
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.bars)
        layout.addWidget(self.flagged_only)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(QLabel("Warnings (each once, with its count)"))
        layout.addWidget(self.warnings)
        layout.addWidget(self.csv_button)
        self.interfaces.itemSelectionChanged.connect(self._select_interface)
        self.cells.itemSelectionChanged.connect(self._select_cell)
        self.flagged_only.toggled.connect(lambda *_: self.show_run(self.run_id))
        self.csv_button.clicked.connect(self._export_csv)
        self.session.results_changed.connect(self.show_run)
        self.session.out_of_date.connect(lambda run_id: self.show_run(self.run_id))
        self._bars_file: Path | None = None
        self._watched_labels: Any = None
        self.showing = "one mesh"

    @staticmethod
    def _table() -> QTableWidget:
        table = QTableWidget()
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSortingEnabled(True)
        return table

    def show_run(self, run_id: str | None) -> None:
        """Fill the tables and the colour bars from a run's last inference."""
        self.run_id = run_id
        run = RUNS.get(run_id or "")
        if run is None:
            return
        if run.last_inference == "ensemble" and run.ensemble is not None:
            self._show_ensemble(run)
            return
        if run.outcome is None:
            return
        self.showing = "one mesh"
        outcome = run.outcome
        truth_t, truth_p = _truth_of(self.session.displays[run.run_id].source_layer)
        tensions = outcome.tensions or {}
        pressures = outcome.pressures or {}
        factor = _scale_factor(tensions, truth_t) if truth_t else None
        cut = {int(c) for c in run.mesh.cut_cells}
        flagged = self.flagged_only.isChecked()

        columns = ["label 1", "label 2", "tension", "undetermined"]
        if truth_t:
            columns += ["true tension", "relative error (scale-matched)"]
        rows = []
        for (a, b), t in sorted(tensions.items()):
            undetermined = not np.isfinite(t)
            if flagged and not undetermined:
                continue
            row = [int(a), int(b), float(t), "yes" if undetermined else ""]
            if truth_t:
                true = truth_t.get((int(a), int(b)), float("nan"))
                row += [true, abs(t * factor - true) / true if np.isfinite(t) and true else float("nan")]
            rows.append(row)
        self._fill(self.interfaces, columns, rows)

        columns = ["label", "pressure", "cut by the image box", "undetermined"]
        if truth_p:
            columns += ["true pressure"]
        rows = []
        for label, p in sorted(pressures.items()):
            undetermined = not np.isfinite(p)
            if flagged and not (undetermined or int(label) in cut):
                continue
            row = [int(label), float(p), "yes" if int(label) in cut else "", "yes" if undetermined else ""]
            if truth_p:
                row.append(truth_p.get(int(label), float("nan")))
            rows.append(row)
        self._fill(self.cells, columns, rows)

        stale = " — OUT OF DATE: the labels changed; reconstruct again" if run.out_of_date else ""
        text = (
            f"{len(tensions)} interfaces, {len(pressures)} materials; status {outcome.status}"
            f"{': ' + outcome.error if outcome.error else ''}{stale}"
        )
        if truth_t:
            errors = [
                abs(t * factor - truth_t[k]) / truth_t[k]
                for k, t in tensions.items()
                if k in truth_t and np.isfinite(t) and truth_t[k]
            ]
            text += (
                f". True tensions known: median relative error {np.median(errors):.3f} after matching the scale "
                f"(factor {factor:.4g} = mean of true / mean of inferred, the benchmark's convention)"
            )
        _status(self.summary, text, AMBER if run.out_of_date else None)
        self.warnings.setPlainText(
            "\n".join(
                f"{w['category']}: {w['message']}" + (f" ({w['count']} times)" if w["count"] > 1 else "")
                for w in [*run.reconstruct_warnings, *run.infer_warnings]
            )
            or "none"
        )
        self._draw_bars(run)
        self._watch_labels(run)

    def _show_ensemble(self, run: Run) -> None:
        """The seed ensemble's table: its tension, the seed spread, how many seeds found each interface."""
        from napari_foambryo._display import interface_keys

        self.showing = "ensemble"
        result = run.ensemble
        on_mesh = set(interface_keys(run.mesh)[1])
        truth_t, _ = _truth_of(self.session.displays[run.run_id].source_layer)
        factor = _scale_factor(result.tensions, truth_t) if truth_t else None
        n_seeds = len(result.seeds)
        columns = [
            "label 1",
            "label 2",
            "ensemble tension",
            "seed SD",
            "relative seed SD",
            "seeds found",
            "found by fewer than half",
            "on the displayed mesh",
        ]
        if truth_t:
            columns += ["true tension", "relative error (scale-matched)"]
        rows = []
        for key in sorted(result.tensions):
            t = result.tensions[key]
            few = bool(result.found_by_fewer_than_half_the_seeds[key])
            shown = key in on_mesh
            if self.flagged_only.isChecked() and not (few or not shown or not np.isfinite(t)):
                continue
            row = [
                int(key[0]),
                int(key[1]),
                float(t),
                float(result.tension_seed_sd[key]),
                float(result.relative_tension_seed_sd[key]),
                f"{result.n_seeds_found[key]} of {n_seeds}",
                "yes" if few else "",
                "" if shown else "no (table only)",
            ]
            if truth_t:
                true = truth_t.get(key, float("nan"))
                row += [true, abs(t * factor - true) / true if np.isfinite(t) and true else float("nan")]
            rows.append(row)
        self._fill(self.interfaces, columns, rows)
        self._fill(self.cells, ["a seed ensemble has no pressures: no single mesh carries them"], [])
        self.tabs.setCurrentWidget(self.interfaces)
        flagged = sum(result.found_by_fewer_than_half_the_seeds.values())
        stale = " — OUT OF DATE: the labels changed; reconstruct again" if run.out_of_date else ""
        text = (
            f"Seed ensemble of {n_seeds} seeds ({result.combine}, {result.mode.name}): {len(result.tensions)} "
            f"interfaces, {flagged} found by fewer than half of the seeds; shown on seed 42's mesh. The seed SD "
            f"is a lower bound on the error, not an error bar.{stale}"
        )
        if truth_t:
            errors = [
                abs(t * factor - truth_t[k]) / truth_t[k]
                for k, t in result.tensions.items()
                if k in truth_t and np.isfinite(t) and truth_t[k]
            ]
            text += f" True tensions known: median relative error {np.median(errors):.3f} after matching the scale."
        _status(self.summary, text, AMBER if run.out_of_date else None)
        self.warnings.setPlainText(
            "\n".join(
                f"{w['category']}: {w['message']}" + (f" ({w['count']} times)" if w["count"] > 1 else "")
                for w in [*run.reconstruct_warnings, *run.ensemble_warnings]
            )
            or "none"
        )
        self._draw_bars(run)
        self._watch_labels(run)

    @staticmethod
    def _fill(table: QTableWidget, columns: list[str], rows: list[list]) -> None:
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem()
                if isinstance(value, float):
                    item.setData(Qt.ItemDataRole.DisplayRole, float(f"{value:.4g}") if np.isfinite(value) else "nan")
                    item.setData(Qt.ItemDataRole.UserRole, value)
                elif isinstance(value, int):
                    item.setData(Qt.ItemDataRole.DisplayRole, value)
                else:
                    item.setText(str(value))
                table.setItem(r, c, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def _draw_bars(self, run: Run) -> None:
        import tempfile

        if self.showing == "ensemble":
            tensions, pressures = run.ensemble.tensions, None
        else:
            tensions = None if run.outcome is None else run.outcome.tensions
            pressures = None if run.outcome is None else run.outcome.pressures
        if tensions is None or not any(np.isfinite(v) for v in tensions.values()):
            self.bars.clear()
            return
        if self._bars_file is None:
            self._bars_folder = tempfile.TemporaryDirectory(prefix="napari_foambryo_bars_")
            self._bars_file = Path(self._bars_folder.name) / "colour_bars.png"
        save_colour_bars(run.mesh, tensions, pressures, self._bars_file)
        pixmap = QPixmap(str(self._bars_file))
        self.bars.setPixmap(pixmap.scaledToWidth(min(420, pixmap.width()), Qt.TransformationMode.SmoothTransformation))

    def _layer(self, kind: str) -> Any:  # noqa: ANN401
        display = self.session.displays.get(self.run_id or "")
        if display is None:
            return None
        for layer in display.layers:
            if layer.metadata.get("foambryo", {}).get("kind") == kind and layer in self.viewer.layers:
                return layer
        return None

    def _select_interface(self) -> None:
        layer = self._layer("ensemble tensions" if self.showing == "ensemble" else "tensions")
        rows = self.interfaces.selectionModel().selectedRows()
        if layer is None:
            return
        if not rows:
            highlight(self.viewer, layer, None)
            return
        self._show_only(layer)
        a = int(self.interfaces.item(rows[0].row(), 0).data(Qt.ItemDataRole.DisplayRole))
        b = int(self.interfaces.item(rows[0].row(), 1).data(Qt.ItemDataRole.DisplayRole))
        keys = layer.metadata["foambryo"]["interfaces"]
        if (a, b) in keys:
            highlight(self.viewer, layer, keys.index((a, b)))

    def _select_cell(self) -> None:
        layer = self._layer("pressures")
        rows = self.cells.selectionModel().selectedRows()
        if layer is None:
            return
        if not rows:
            highlight(self.viewer, layer, None)
            return
        self._show_only(layer)
        highlight(self.viewer, layer, int(self.cells.item(rows[0].row(), 0).data(Qt.ItemDataRole.DisplayRole)))

    def _show_only(self, layer: Any) -> None:  # noqa: ANN401
        """Show `layer` and hide the run's other result layers (they share the geometry)."""
        for other in self.session.displays[self.run_id].layers:
            if other.metadata.get("foambryo", {}).get("kind") in (
                "cells",
                "tensions",
                "pressures",
                "ensemble tensions",
            ):
                other.visible = other is layer

    def select_cell_row(self, label: int) -> None:
        """Select the row of cell `label` in the Cells tab (a click on the Labels layer does this)."""
        for row in range(self.cells.rowCount()):
            if int(self.cells.item(row, 0).data(Qt.ItemDataRole.DisplayRole)) == label:
                self.tabs.setCurrentWidget(self.cells)
                self.cells.selectRow(row)
                return

    def _watch_labels(self, run: Run) -> None:
        labels = self.session.displays[run.run_id].source_layer
        if labels is None or labels is self._watched_labels:
            return
        self._watched_labels = labels

        def clicked(layer: Any, event: Any) -> None:  # noqa: ANN401
            value = layer.get_value(
                event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True
            )
            if isinstance(value, tuple):
                value = value[-1]
            if value is not None and int(value) > 0:
                self.select_cell_row(int(value))

        labels.mouse_drag_callbacks.append(clicked)

    def _export_csv(self) -> None:
        run = RUNS.get(self.run_id or "")
        if run is None or (run.outcome is None and run.ensemble is None):
            return
        cells = self.tabs.currentWidget() is self.cells
        name = tables.PRESSURE_TABLE if cells else tables.TENSION_TABLE
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", name, "CSV (*.csv)")
        if path:
            self.write_csv(Path(path), cells=cells)

    def write_csv(self, path: Path, *, cells: bool) -> None:
        """Write the current run's tensions or pressures in the command's format (exact floats)."""
        run = RUNS[self.run_id or ""]
        if self.showing == "ensemble":
            if not cells:
                tables.write_ensemble_tension_table(path, run.ensemble)
            return
        if cells and run.outcome.pressures is not None:
            tables.write_pressure_table(path, run.outcome.pressures)
        elif not cells and run.outcome.tensions is not None:
            tables.write_tension_table(path, run.outcome.tensions)


def _truth_of(layer: Any) -> tuple[dict, dict]:  # noqa: ANN401
    info = getattr(layer, "metadata", {}).get("foambryo", {}) if layer is not None else {}
    tensions = {(int(a), int(b)): float(t) for a, b, t in info.get("true_tensions", [])}
    pressures = {int(k): float(p) for k, p in info.get("true_pressures", [])}
    return tensions, pressures


def _scale_factor(inferred: dict, truth: dict) -> float:
    """`mean(truth) / mean(inferred)` over the shared, finite interfaces: the benchmark's scale matching."""
    shared = [k for k in truth if k in inferred and np.isfinite(inferred[k])]
    if not shared:
        return 1.0
    mean_inferred = float(np.mean([inferred[k] for k in shared]))
    return float(np.mean([truth[k] for k in shared])) / mean_inferred if mean_inferred else 1.0


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------
class ExportWidget(QWidget):
    """A folder and one button: the command's files, and the command lines that reproduce them."""

    def __init__(self, napari_viewer: Any) -> None:  # noqa: ANN401
        """No default folder: the user chooses where files go."""
        super().__init__()
        self.viewer = napari_viewer
        self.session = session_for(napari_viewer)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("choose a folder")
        browse = QPushButton("Browse…")
        self.include_mask = QCheckBox("include the mask (mask.npz)")
        self.overwrite = QCheckBox("replace an earlier export in this folder")
        self.export_button = QPushButton("Export")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row = QHBoxLayout()
        row.addWidget(self.folder)
        row.addWidget(browse)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.include_mask)
        layout.addWidget(self.overwrite)
        layout.addWidget(self.export_button)
        layout.addWidget(self.status)
        layout.addStretch(1)
        browse.clicked.connect(self._browse)
        self.export_button.clicked.connect(self.export)
        self.session.results_changed.connect(self._refresh)
        self.session.out_of_date.connect(self._refresh)
        self.folder.textChanged.connect(self._refresh)
        self._refresh()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export folder")
        if folder:
            self.folder.setText(folder)

    def _refresh(self, *_args: Any) -> None:  # noqa: ANN401
        run = self.session.run()
        if run is None or (run.outcome is None and run.ensemble is None):
            self.export_button.setEnabled(False)
            self.export_button.setText("Export (run Infer first)")
        elif run.out_of_date:
            self.export_button.setEnabled(False)
            self.export_button.setText("Export (the labels changed: reconstruct again)")
        else:
            self.export_button.setEnabled(bool(self.folder.text().strip()))
            self.export_button.setText("Export")

    def export(self) -> list[str]:
        """Write the current run's files; returns their names (empty when refused)."""
        run = self.session.run()
        if run is None:
            return []
        display = self.session.displays.get(run.run_id)
        if display is not None and display.source_layer is not None and run.source is not None:
            try:
                current = prepare_mask(display.source_layer.data, multiscale=display.source_layer.multiscale).mask
            except SegmentationMaskError:
                current = None
            changed = current is None or (
                array_sha256(current) != run.source.sha256 and array_sha256(current) != array_sha256(run.mask)
            )
            if changed:
                self.session.mark_out_of_date(run.run_id)
        try:
            written = export_run(
                run,
                Path(self.folder.text().strip()).expanduser(),
                include_mask=self.include_mask.isChecked(),
                overwrite=self.overwrite.isChecked(),
            )
        except (ValueError, FileExistsError, OSError) as error:
            _status(self.status, str(error), RED)
            return []
        commands = "\n".join(run.equivalent_commands())
        _status(
            self.status,
            f"Wrote {', '.join(written)} in {self.folder.text().strip()}.\n"
            f"The same numbers from the command line (in that folder):\n{commands}",
        )
        return written
