"""The state the dock widgets share in one viewer: the runs, which one is current, and their layers.

The widgets do not call each other: Reconstruct adds a run and says so, Infer computes on the current run and says
so, Results, Export and View listen; View redraws a run's layers (explode, cut) and says so, and Results selects its
row again. A change of a run's source Labels layer marks the run out of date (its layers are
renamed `… [out of date]` and Export refuses it) until it is reconstructed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from napari_foambryo._layers import RUNS

if TYPE_CHECKING:
    from napari_foambryo._runs import Run

__all__ = ["RunDisplay", "Session", "session_for"]

OUT_OF_DATE_SUFFIX = " [out of date]"
_SESSIONS: dict[int, Session] = {}


@dataclass
class RunDisplay:
    """How a run is shown: the base of its layers' names, its source layer, the spacing it was reconstructed with.

    `explode` is the View dock's explode coefficient for this run (0 = off), `cut` its cut (`_layers.CutPlane`) or
    `None`: layers added later (another Infer) are drawn with both.
    """

    base_name: str
    source_layer: Any = None
    spacing_used: tuple[float, float, float] | None = None
    layers: list = field(default_factory=list)
    explode: float = 0.0
    cut: Any = None


class Session:
    """One viewer's runs and signals."""

    run_added = Signal(str)
    results_changed = Signal(str)
    out_of_date = Signal(str)
    display_changed = Signal(str)

    def __init__(self, viewer: Any) -> None:  # noqa: ANN401 - napari.Viewer or ViewerModel
        """Start with no run."""
        self.viewer = viewer
        self.current: str | None = None
        self.displays: dict[str, RunDisplay] = {}
        self.order: list[str] = []
        #: The Infer dock's current options (a callable returning `InferenceOptions`), so that Export can say when
        #: they differ from those of the inference it exports.
        self.infer_form_options: Any = None

    def run(self, run_id: str | None = None) -> Run | None:
        """The run `run_id`, or the current one."""
        return RUNS.get(run_id or self.current or "")

    def add(self, run: Run, display: RunDisplay) -> None:
        """Make `run` the current run, replacing an earlier run with the same layer names."""
        for other, shown in list(self.displays.items()):
            if shown.base_name == display.base_name:
                self._remove_layers(shown)
                self.displays.pop(other)
                self.order.remove(other)
        self.displays[run.run_id] = display
        self.order.append(run.run_id)
        self.current = run.run_id
        if display.source_layer is not None:
            self.watch(run.run_id, display.source_layer)
        self.run_added.emit(run.run_id)

    def _remove_layers(self, display: RunDisplay) -> None:
        for layer in display.layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        display.layers.clear()

    def watch(self, run_id: str, layer: Any) -> None:  # noqa: ANN401
        """Mark the run out of date when `layer`'s data changes."""

        def mark(_event: Any = None) -> None:  # noqa: ANN401
            self.mark_out_of_date(run_id)

        for name in ("data", "paint", "labels_update", "set_data"):
            emitter = getattr(layer.events, name, None)
            if emitter is not None:
                emitter.connect(mark)

    def mark_out_of_date(self, run_id: str) -> None:
        """The source labels changed: rename the run's layers and refuse to export it."""
        run = RUNS.get(run_id)
        if run is None or run.out_of_date:
            return
        run.out_of_date = True
        for layer in self.displays[run_id].layers:
            if not layer.name.endswith(OUT_OF_DATE_SUFFIX):
                layer.name = layer.name + OUT_OF_DATE_SUFFIX
        self.out_of_date.emit(run_id)


def _unwrapped(viewer: Any) -> Any:  # noqa: ANN401
    """The viewer itself: napari hands each plugin widget its own proxy (`PublicOnlyProxy`) around it."""
    while hasattr(viewer, "__wrapped__"):
        viewer = viewer.__wrapped__
    return viewer


def session_for(viewer: Any) -> Session:  # noqa: ANN401
    """The session of a viewer (created on first use), the same for every proxy of that viewer."""
    viewer = _unwrapped(viewer)
    key = id(viewer)
    session = _SESSIONS.get(key)
    if session is None or session.viewer is not viewer:
        session = _SESSIONS[key] = Session(viewer)
    return session
