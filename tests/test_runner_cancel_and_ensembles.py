"""The child-process runner (Cancel, refusals, progress) and seed ensembles: gates G4 and G5 (tier T1).

* Cancel stops the child **and the processes it started** (a seed ensemble's workers), and returns nothing.
* A refusal in the child arrives as `ChildFailedError` with the library's message; progress calls arrive in order.
* G4: a seed ensemble run by the plugin's runner is bitwise the library's, with 1 and 2 workers.
* G5: on a cropped case of benchmark size (case 000 cut by the image box), the in-memory mesh and the `.rec`
  route (crop faces written as the last rows, `foambryo infer`'s reading) give bitwise the same numbers.
* The ensemble's export is byte for byte `foambryo infer-over-seeds`' tables.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
from foambryo import (
    PressureComputationMethod,
    TensionComputationMethod,
    cli,
    dcel_mesh_from_file,
    infer_forces,
    infer_tensions_over_seeds,
    reconstruct_mask,
)

from napari_foambryo._masks import array_sha256
from napari_foambryo._runner import ChildCancelledError, ChildFailedError, run_in_child
from napari_foambryo._runs import EnsembleOptions, Run, SourceMask, export_run
from napari_foambryo._sample_data import benchmark_case_000
from tests.conftest import quietly, same_floats, sha


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_cancel_stops_the_child_and_its_own_processes(tmp_path):
    pid_file = tmp_path / "pids.txt"
    cancel = threading.Event()
    outcome = {}

    def run() -> None:
        try:
            run_in_child(
                "tests.slow_targets:sleep_with_a_grandchild", {"pid_file": str(pid_file)}, should_cancel=cancel.is_set
            )
        except ChildCancelledError:
            outcome["cancelled"] = True

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 60
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    child, grandchild = (int(v) for v in pid_file.read_text().split())
    assert not _gone(child) and not _gone(grandchild)
    started = time.monotonic()
    cancel.set()
    thread.join(30)
    assert outcome == {"cancelled": True}
    assert time.monotonic() - started < 10
    deadline = time.monotonic() + 10
    while not (_gone(child) and _gone(grandchild)) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _gone(child) and _gone(grandchild)


def test_a_refusal_and_progress_cross_the_process_boundary():
    with pytest.raises(ChildFailedError) as raised:
        run_in_child("tests.slow_targets:refuse", {"message": "no formulation for this"})
    assert (raised.value.kind, raised.value.message) == ("ValueError", "no formulation for this")
    assert "Traceback" in raised.value.child_traceback
    calls = []
    result = run_in_child("tests.slow_targets:report_progress", {"n": 4}, on_progress=lambda *a: calls.append(a))
    assert result.value == 4
    assert calls == [(1, 4, 0), (2, 4, 1), (3, 4, 2), (4, 4, 3)]


def _same_ensemble(a, b) -> bool:
    return (
        a.seeds == b.seeds
        and same_floats(a.tensions, b.tensions)
        and same_floats(a.tension_seed_sd, b.tension_seed_sd)
        and same_floats(a.relative_tension_seed_sd, b.relative_tension_seed_sd)
        and a.n_seeds_found == b.n_seeds_found
        and a.found_by_fewer_than_half_the_seeds == b.found_by_fewer_than_half_the_seeds
        and all(same_floats(a.per_seed_tensions[s], b.per_seed_tensions[s]) for s in a.seeds)
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_a_seed_ensemble_from_the_child_is_the_librarys(samples, workers):
    mask, spacing = samples["anisotropic"]
    options = EnsembleOptions(3, "young-dupre", "tensions", "chord", workers)
    calls = []
    child = run_in_child(
        "foambryo.seed_ensembles:infer_tensions_over_seeds",
        options.library_kwargs(mask, spacing, 3),
        on_progress=lambda *a: calls.append(a),
    )
    library = quietly(infer_tensions_over_seeds, mask, range(3), spacing=spacing, max_workers=workers)
    assert _same_ensemble(child.value, library)
    assert [c[0] for c in calls] == [1, 2, 3]


def test_the_in_memory_and_rec_routes_agree_on_a_cropped_benchmark_case(tmp_path):
    ((mask, _, _),) = benchmark_case_000()
    cut = np.ascontiguousarray(mask[:, :110, :])  # the image box now cuts cells
    reconstruction = quietly(reconstruct_mask, cut, (1.0, 1.0, 1.0), spacing_unit="voxel")
    in_memory = reconstruction.mesh()
    assert in_memory.crop_faces.any()
    reconstruction.write_rec(tmp_path / "cut.rec")
    from_file = quietly(dcel_mesh_from_file, tmp_path / "cut.rec")
    assert int(from_file.crop_faces.sum()) == int(in_memory.crop_faces.sum())
    for tension in (
        TensionComputationMethod.YoungDupre,
        TensionComputationMethod.ForceBalance,
        TensionComputationMethod.Lami,
        TensionComputationMethod.Variational,
    ):
        here = quietly(
            infer_forces, in_memory, mode_tension=tension, mode_pressure=PressureComputationMethod.Variational
        )
        there = quietly(
            infer_forces, from_file, mode_tension=tension, mode_pressure=PressureComputationMethod.Variational
        )
        assert same_floats(here[0], there[0]), tension
        assert same_floats(here[1], there[1]), tension
    assert sha(np.sort(in_memory.f, axis=0)) == sha(np.sort(from_file.f, axis=0))


def test_the_ensemble_export_is_what_infer_over_seeds_writes(samples, tmp_path, capsys):
    mask, spacing = samples["cubic"]
    reconstruction = quietly(reconstruct_mask, mask, spacing)
    options = EnsembleOptions(2, "young-dupre", "tensions", "chord", 1)
    source = SourceMask("sample", mask.shape, str(mask.dtype), array_sha256(mask), spacing, "um", "this layer")
    run = Run(
        "ensemble-test",
        reconstruction.mesh(),
        source=source,
        mask=mask,
        reconstruction=reconstruction,
        rec_bytes=reconstruction.rec_bytes,
        ensemble_options=options,
    )
    child = run_in_child("foambryo.seed_ensembles:infer_tensions_over_seeds", options.library_kwargs(mask, spacing, 3))
    run.ensemble, run.ensemble_warnings, run.ensemble_seconds = child.value, child.warnings_summary, child.seconds
    written = export_run(run, tmp_path / "exported", include_mask=True)
    assert "seed-ensemble/tensions.csv" in written and "seed-ensemble/tensions_per_seed.csv" in written
    record = json.loads((tmp_path / "exported" / "run.json").read_text())
    line = record["equivalent_commands"][-1]
    assert line.startswith("foambryo infer-over-seeds mask.npz -o seed-ensemble")
    folder = tmp_path / "command"
    folder.mkdir()
    (folder / "mask.npz").write_bytes((tmp_path / "exported" / "mask.npz").read_bytes())
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        assert cli.main(shlex.split(line)[1:]) == cli.EXIT_OK
    finally:
        os.chdir(cwd)
    capsys.readouterr()
    for name in ("tensions.csv", "tensions_per_seed.csv"):
        ours = (tmp_path / "exported" / "seed-ensemble" / name).read_bytes()
        assert ours == (folder / "seed-ensemble" / name).read_bytes(), name
    ensemble_record = json.loads((tmp_path / "exported" / "seed-ensemble" / "run.json").read_text())
    command_record = json.loads((folder / "seed-ensemble" / "run.json").read_text())
    for key in ("schema", "options", "seeds", "library_call", "results", "units", "status"):
        assert ensemble_record[key] == command_record[key], key
    assert Path(tmp_path / "exported" / "mesh.rec").read_bytes() == reconstruction.rec_bytes
