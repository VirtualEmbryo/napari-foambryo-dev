"""Tier T1/T2: the `.rec` reader and writer, the sample data, and the manifest.

* The reader opens a `foambryo mesh` file as layers, with the mesh foambryo's own reader builds (so Infer on it is
  `foambryo infer` on the file); it declines other files.
* The writer writes a plugin layer's mesh file byte for byte, and declines other layers.
* Case 000 ships its mask and true values; the generated foams are what their names say.
"""

from __future__ import annotations

import numpy as np
from foambryo import dcel_mesh_from_file, infer_forces, reconstruct_mask

from napari_foambryo._io import get_reader, write_rec
from napari_foambryo._layers import RUNS
from napari_foambryo._sample_data import benchmark_case_000, foam_anisotropic, foam_cropped, foam_cubic, true_values
from tests.conftest import quietly, same_floats, sha


def test_the_reader_opens_a_command_mesh_file_and_declines_others(samples, tmp_path):
    mask, spacing = samples["cropped"]
    reconstruction = quietly(reconstruct_mask, mask, spacing)
    path = tmp_path / "tissue.rec"
    reconstruction.write_rec(path)
    assert get_reader(str(tmp_path / "image.tif")) is None
    layers = get_reader(str(path))(str(path))
    assert [meta["name"] for _, meta, _ in layers] == ["tissue cells", "tissue crop faces"]
    run = RUNS[layers[0][1]["metadata"]["foambryo"]["run_id"]]
    from_file = quietly(dcel_mesh_from_file, path)
    assert sha(run.mesh.v) == sha(from_file.v) and sha(run.mesh.f) == sha(from_file.f)
    assert run.rec_bytes == path.read_bytes()
    ours, theirs = quietly(infer_forces, run.mesh), quietly(infer_forces, from_file)
    assert same_floats(ours[0], theirs[0]) and same_floats(ours[1], theirs[1])

    written = write_rec(str(tmp_path / "copy"), layers[0][0], layers[0][1])
    assert written == [str(tmp_path / "copy.rec")]
    assert (tmp_path / "copy.rec").read_bytes() == path.read_bytes()
    assert write_rec(str(tmp_path / "other.rec"), None, {"metadata": {}}) == []


def test_case_000_ships_its_mask_and_true_values():
    ((mask, meta, kind),) = benchmark_case_000()
    assert kind == "labels"
    assert mask.shape == (194, 200, 199) and mask.dtype == np.uint8
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2, 3, 4, 5, 6]
    tensions, pressures = true_values()
    assert len(tensions) == 18 and len(pressures) == 7
    assert all(0 <= a < b <= 6 for a, b in tensions)
    info = meta["metadata"]["foambryo"]
    assert info["spacing"] == (1.0, 1.0, 1.0) and info["spacing_unit"] == "voxel"
    assert len(info["true_tensions"]) == 18


def test_the_generated_foams_are_what_their_names_say():
    ((cubic, cubic_meta, _),) = foam_cubic()
    ((anisotropic, anisotropic_meta, _),) = foam_anisotropic()
    ((cropped, _, _),) = foam_cropped()
    assert len(np.unique(cubic)) == 13  # 12 cells and the medium
    assert np.array_equal(anisotropic, cubic[::2])
    assert anisotropic_meta["scale"][0] == 2 * cubic_meta["scale"][0]
    assert np.array_equal(cropped, cubic[20:])
    assert cropped[0].any() and not cubic[0].any()  # the cut reaches the cells; the whole foam does not


def test_the_manifest_is_valid():
    from npe2 import PluginManifest

    manifest = PluginManifest.from_distribution("napari-foambryo")
    assert {w.display_name for w in manifest.contributions.widgets} == {
        "Reconstruct",
        "Infer",
        "Results",
        "Export",
        "View",
    }
    assert len(manifest.contributions.sample_data) == 4
