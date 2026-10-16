"""Tier T1: Export writes the `foambryo` command's own files.

* `mesh.rec` is byte for byte what `foambryo mesh` writes from the exported `mask.npz`;
* `foambryo infer mesh.rec` writes `tensions.csv` and `pressures.csv` byte for byte as exported;
* `run.json` has every key of schema 1 with the command's values (except `command`, `arguments`, `date`,
  `elapsed_seconds`, `versions`, `warnings`, `solves`), plus `source_mask` and `equivalent_commands`;
* the equivalent command lines, run in the folder, reproduce it.
"""

from __future__ import annotations

import json
import os
import shlex

import pytest
from foambryo import cli
from foambryo import results as tables

from napari_foambryo._masks import array_sha256
from napari_foambryo._runner import run_in_child
from napari_foambryo._runs import InferenceOptions, Run, SourceMask, export_run, infer_on_mesh

SAME_VALUES = ("schema", "options", "library_call", "mesh", "results", "units", "status", "error")


def _run_for(mask, spacing, options: InferenceOptions) -> Run:
    child = run_in_child(
        "foambryo.io:reconstruct_mask", {"segmentation_mask": mask, "spacing": spacing, "spacing_unit": "um"}
    )
    reconstruction = child.value
    source = SourceMask("sample", tuple(mask.shape), str(mask.dtype), array_sha256(mask), spacing, "um", "this layer")
    run = Run(
        "run-test",
        reconstruction.mesh(),
        source=source,
        mask=mask,
        reconstruction=reconstruction,
        rec_bytes=reconstruction.rec_bytes,
    )
    run.options = options
    run.outcome, run.infer_warnings, run.infer_seconds = infer_on_mesh(run.mesh, options)
    run.infer_date = "2026-10-12T00:00:00+00:00"
    return run


@pytest.mark.parametrize("name", ["cubic", "anisotropic", "cropped"])
def test_the_exported_folder_is_what_the_command_writes(samples, tmp_path, name, capsys):
    mask, spacing = samples[name]
    options = InferenceOptions("young-dupre", "variational", 1.0, 0.0, "auto", "none")
    run = _run_for(mask, spacing, options)
    exported = tmp_path / "exported"
    written = export_run(run, exported, include_mask=True)
    assert set(written) == {"mesh.rec", "tensions.csv", "pressures.csv", "colour_bars.png", "mask.npz", "run.json"}

    record = json.loads((exported / "run.json").read_text())
    commands = record["equivalent_commands"]
    assert commands[0].startswith("foambryo mesh mask.npz -o mesh.rec --spacing ")
    command_folder = tmp_path / "command"
    command_folder.mkdir()
    (command_folder / "mask.npz").write_bytes((exported / "mask.npz").read_bytes())
    cwd = os.getcwd()
    try:
        os.chdir(command_folder)
        for line in commands:
            arguments = shlex.split(line)
            assert arguments[0] == "foambryo"
            assert cli.main(arguments[1:]) == cli.EXIT_OK
    finally:
        os.chdir(cwd)
    capsys.readouterr()
    assert (command_folder / "mesh.rec").read_bytes() == (exported / "mesh.rec").read_bytes()
    for table in ("tensions.csv", "pressures.csv"):
        assert (command_folder / table).read_bytes() == (exported / table).read_bytes(), table

    command_record = json.loads((command_folder / "run.json").read_text())
    for key in command_record:
        assert key in record, key
    for key in SAME_VALUES:
        assert record[key] == command_record[key], key
    assert record["input"]["sha256"] == command_record["input"]["sha256"]
    assert record["source_mask"]["sha256"] == array_sha256(mask)
    assert record["versions"]["napari-foambryo"]
    assert record["command"] == "napari-foambryo"


def test_export_refuses_an_out_of_date_run_and_an_existing_folder(samples, tmp_path):
    mask, spacing = samples["cubic"]
    run = _run_for(mask, spacing, InferenceOptions("young-dupre", "variational", 1.0, 0.0, "auto", "none"))
    assert "mask.npz" not in export_run(run, tmp_path)
    assert "not exported" in json.loads((tmp_path / tables.RUN_RECORD).read_text())["equivalent_commands_note"]
    with pytest.raises(FileExistsError):
        export_run(run, tmp_path)
    export_run(run, tmp_path, overwrite=True)
    run.out_of_date = True
    with pytest.raises(ValueError, match="labels changed"):
        export_run(run, tmp_path / "again")
