"""A run of the plugin (a mask, its reconstruction, its inference) and the folder Export writes from it.

No Qt and no napari: the widgets fill a `Run`, and `export_run` writes the `foambryo` command's own files with
foambryo's own writers (`foambryo.results`), so that the folder can be reproduced, and checked, without napari:

* `mesh.rec`: the reconstruction's `.rec` bytes, which are `foambryo mesh`'s file for the same mask and options;
* `tensions.csv`, `pressures.csv` (and `spectra.npz` with the spectrum diagnostics): exact floats;
* `run.json`: schema 1 of the command's run record, plus `source_mask`, `equivalent_commands`, and napari's and the
  plugin's versions;
* `colour_bars.png`: `foambryo.colour_scales.save_colour_bars` on the values shown;
* `mask.npz` (on request, and always when the medium label was swapped): what `foambryo mesh` reads.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from foambryo import infer_tensions_then_pressures
from foambryo import results as tables
from foambryo.colour_scales import save_colour_bars

if TYPE_CHECKING:
    from foambryo.dcel import DcelData
    from foambryo.force_inference import InferenceOutcome
    from foambryo.io import MaskReconstruction

__all__ = ["EXPORT_OUTPUTS", "InferenceOptions", "Run", "SourceMask", "export_run", "infer_on_mesh"]

PLUGIN = "napari-foambryo"
MESH_FILE = "mesh.rec"
MASK_FILE = "mask.npz"
COLOUR_BARS = "colour_bars.png"
#: Every file Export may write, and so the only ones it replaces.
EXPORT_OUTPUTS = (MESH_FILE, *tables.INFER_OUTPUTS, COLOUR_BARS, MASK_FILE)


@dataclass(frozen=True)
class SourceMask:
    """Where the reconstructed mask came from, as `run.json`'s `source_mask` records it."""

    layer_name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    spacing: tuple[float, float, float]
    spacing_unit: str
    spacing_from: str
    medium_label: int = 0
    notes: tuple[str, ...] = ()
    dropped_leading_axis: bool = False
    multiscale_level: int | None = None
    origin: dict[str, Any] = field(default_factory=dict)
    mask_report: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        """The `source_mask` entry of `run.json`."""
        return {
            "layer_name": self.layer_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
            "spacing": list(self.spacing),
            "spacing_unit": self.spacing_unit,
            "spacing_from": self.spacing_from,
            "medium_label": self.medium_label,
            "medium_swapped": self.medium_label != 0,
            "dropped_leading_axis": self.dropped_leading_axis,
            "multiscale_level": self.multiscale_level,
            "notes": list(self.notes),
            "origin": self.origin,
            "mask_report": self.mask_report,
        }


@dataclass(frozen=True)
class InferenceOptions:
    """The Infer widget's options, by the command's spellings (`foambryo.results.option_name`)."""

    tension_method: str
    pressure_method: str
    mean_tension: float
    base_pressure: float
    solver: str
    diagnostics: str

    def as_record(self) -> dict[str, Any]:
        """The `options` entry of `run.json`, as `foambryo infer` writes it."""
        return {
            "tension_method": self.tension_method,
            "pressure_method": self.pressure_method,
            "mean_tension": self.mean_tension,
            "base_pressure": self.base_pressure,
            "solver": self.solver,
            "diagnostics": self.diagnostics,
        }

    def command_arguments(self) -> list[str]:
        """The `foambryo infer` options that give the same numbers."""
        return [
            "--tension-method",
            self.tension_method,
            "--pressure-method",
            self.pressure_method,
            "--mean-tension",
            repr(self.mean_tension),
            "--base-pressure",
            repr(self.base_pressure),
            "--solver",
            self.solver,
            "--diagnostics",
            self.diagnostics,
        ]


def infer_on_mesh(mesh: DcelData, options: InferenceOptions) -> tuple[InferenceOutcome, list[dict], float]:
    """`foambryo.infer_tensions_then_pressures` with the widget's options; its warnings counted; its time.

    Returns:
        tuple: the outcome, the warnings (each once with its count), the seconds.
    """
    collector = tables.WarningCollector()
    started = time.perf_counter()
    with collector:
        outcome = infer_tensions_then_pressures(
            mesh,
            options.mean_tension,
            options.base_pressure,
            tables.TENSION_METHODS[options.tension_method],
            tables.PRESSURE_METHODS[options.pressure_method],
            solver=options.solver,
            diagnostics=tables.DIAGNOSTICS[options.diagnostics],
        )
    return outcome, collector.summary, time.perf_counter() - started


@dataclass
class Run:
    """One mask's reconstruction and inference in the plugin.

    Attributes:
        run_id: a short identifier, unique in the session, written in the layers' metadata.
        mesh: the mesh inferred on.
        source: where the mask came from, or `None` for a mesh opened from a `.rec` file.
        mask: the mask that was reconstructed (after any medium swap), or `None`.
        reconstruction: the reconstruction, or `None` for a mesh opened from a file.
        rec_bytes: the mesh file's bytes (the reconstruction's, or the opened file's).
        rec_name: the opened file's name, for a mesh opened from a file.
        reconstruct_warnings: the reconstruction's warnings.
        reconstruct_seconds, peak_rss_bytes: the reconstruction's time and peak memory.
        options, outcome, infer_warnings, infer_seconds, infer_date: the last inference.
        out_of_date: the source layer changed after the reconstruction.
    """

    run_id: str
    mesh: DcelData
    source: SourceMask | None = None
    mask: np.ndarray | None = None
    reconstruction: MaskReconstruction | None = None
    rec_bytes: bytes = b""
    rec_name: str | None = None
    reconstruct_warnings: list = field(default_factory=list)
    reconstruct_seconds: float = 0.0
    peak_rss_bytes: int = 0
    options: InferenceOptions | None = None
    outcome: InferenceOutcome | None = None
    infer_warnings: list = field(default_factory=list)
    infer_seconds: float = 0.0
    infer_date: str = ""
    out_of_date: bool = False

    def equivalent_commands(self) -> list[str]:
        """The command lines that reproduce the exported folder (run in the folder)."""
        commands = []
        if self.source is not None and self.reconstruction is not None:
            spacing = " ".join(repr(float(v)) for v in self.source.spacing)
            unit = shlex.quote(self.source.spacing_unit)
            commands.append(
                f"foambryo mesh {MASK_FILE} -o {MESH_FILE} --spacing {spacing} --spacing-unit {unit} "
                f"--min-distance {self.reconstruction.min_distance}"
            )
        if self.options is not None:
            commands.append(" ".join(["foambryo infer", MESH_FILE, "-o", ".", *self.options.command_arguments()]))
        return commands


def export_run(run: Run, folder: Path, *, include_mask: bool = False, overwrite: bool = False) -> list[str]:
    """Write the run's files in `folder` (created if missing); returns their names.

    Raises:
        FileExistsError: if the folder holds an earlier export and `overwrite` is false.
        ValueError: if the run is out of date, or holds no inference.
    """
    if run.out_of_date:
        message = "the labels changed after the reconstruction: reconstruct again before exporting"
        raise ValueError(message)
    if run.outcome is None or run.options is None:
        message = "nothing to export yet: run Infer first"
        raise ValueError(message)
    folder = Path(folder)
    existing = [name for name in EXPORT_OUTPUTS if (folder / name).exists()]
    if existing and not overwrite:
        message = f"{folder} already holds {', '.join(existing)}"
        raise FileExistsError(message)
    folder.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (folder / name).unlink()

    outcome, options = run.outcome, run.options
    outputs = [MESH_FILE]
    (folder / MESH_FILE).write_bytes(run.rec_bytes)
    write_mask = run.mask is not None and (include_mask or (run.source is not None and run.source.medium_label != 0))
    if write_mask:
        np.savez_compressed(
            folder / MASK_FILE,
            mask=run.mask,
            spacing=np.array(run.source.spacing),
            spacing_unit=np.array(run.source.spacing_unit),
        )
    if outcome.tensions is not None:
        tables.write_tension_table(folder / tables.TENSION_TABLE, outcome.tensions)
        outputs.append(tables.TENSION_TABLE)
    if outcome.pressures is not None:
        tables.write_pressure_table(folder / tables.PRESSURE_TABLE, outcome.pressures)
        outputs.append(tables.PRESSURE_TABLE)
    if options.diagnostics == "spectrum":
        tables.write_spectra(folder / tables.SPECTRA, [record.singular_values for record in outcome.records])
        outputs.append(tables.SPECTRA)
    if outcome.tensions is not None and any(np.isfinite(v) for v in outcome.tensions.values()):
        save_colour_bars(run.mesh, outcome.tensions, outcome.pressures, folder / COLOUR_BARS)
        outputs.append(COLOUR_BARS)
    if write_mask:
        outputs.append(MASK_FILE)
    outputs.append(tables.RUN_RECORD)

    mesh_path = folder / MESH_FILE
    metadata = tables.read_rec_metadata(mesh_path)
    record = tables.infer_run_record(
        command=PLUGIN,
        arguments=[],
        date=run.infer_date,
        elapsed_seconds=round(run.infer_seconds, 3),
        input_record={
            "path": MESH_FILE,
            "sha256": tables.file_sha256(mesh_path),
            "bytes": mesh_path.stat().st_size,
            "metadata": metadata,
        },
        options=options.as_record(),
        mesh=run.mesh,
        outcome=outcome,
        units=tables.infer_units(options.mean_tension, metadata),
        warnings_summary=run.infer_warnings,
        outputs=outputs,
        versions=tables.library_versions(extra=("napari", PLUGIN)),
    )
    record["source_mask"] = None if run.source is None else run.source.as_record()
    if run.rec_name is not None:
        record["source_mesh"] = {"name": run.rec_name}
    record["reconstruction"] = (
        None
        if run.reconstruction is None
        else {
            "summary": run.reconstruction.summary(),
            "min_distance": run.reconstruction.min_distance,
            "seconds": round(run.reconstruct_seconds, 3),
            "peak_rss_bytes": run.peak_rss_bytes,
            "warnings": run.reconstruct_warnings,
        }
    )
    record["equivalent_commands"] = run.equivalent_commands()
    if run.source is not None and not write_mask:
        record["equivalent_commands_note"] = (
            f"{MASK_FILE} was not exported: the mask is identified by "
            "source_mask.sha256 (tick 'include the mask' to export it)"
        )
    record["exported"] = datetime.now(UTC).isoformat(timespec="seconds")
    tables.write_run_record(folder / tables.RUN_RECORD, record)
    return outputs
