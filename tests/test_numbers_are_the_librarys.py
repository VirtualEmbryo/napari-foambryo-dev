"""Gates G1–G3 (tier T1): the plugin's runner gives bitwise the library's mesh and numbers.

The plugin reconstructs in a child process and infers in a worker thread; these tests run the same runner and
compare, by the bytes of the arrays and of every float, with the library called directly in this process:
`dcel_mesh_from_segmentation_mask(mask, spacing=...)` then `infer_forces`. Cubic voxels (G1), anisotropic voxels
(G2), a tissue cut by the image box (G3), for every tension method and every pressure method the library accepts on
that mesh; the library's refusals are reported with the command's status.
"""

from __future__ import annotations

import pytest
from foambryo import (
    PressureComputationMethod,
    TensionComputationMethod,
    dcel_mesh_from_segmentation_mask,
    infer_forces,
    infer_tensions,
)
from foambryo import results as tables

from napari_foambryo._runner import run_in_child
from napari_foambryo._runs import InferenceOptions, infer_on_mesh
from tests.conftest import quietly, same_floats, sha


@pytest.fixture(scope="module")
def reconstructions(samples):
    """Each sample reconstructed by the plugin's runner (a child process) and by the library in this process."""
    out = {}
    for name, (mask, spacing) in samples.items():
        child = run_in_child(
            "foambryo.io:reconstruct_mask", {"segmentation_mask": mask, "spacing": spacing, "spacing_unit": "um"}
        )
        out[name] = (child, quietly(dcel_mesh_from_segmentation_mask, mask, spacing=spacing))
    return out


@pytest.mark.parametrize("name", ["cubic", "anisotropic", "cropped"])
def test_the_mesh_from_the_child_process_is_the_librarys(reconstructions, name):
    child, library = reconstructions[name]
    mesh = child.value.mesh()
    assert sha(mesh.v) == sha(library.v)
    assert sha(mesh.f) == sha(library.f)
    assert sha(mesh.crop_faces) == sha(library.crop_faces)
    assert mesh.source_spacing == library.source_spacing
    assert child.peak_rss_bytes > 0
    assert (name == "cropped") == bool(mesh.crop_faces.any())


def _options(tension: TensionComputationMethod, pressure: PressureComputationMethod) -> InferenceOptions:
    return InferenceOptions(tables.option_name(tension), tables.option_name(pressure), 1.0, 0.0, "auto", "none")


@pytest.mark.parametrize("name", ["cubic", "anisotropic", "cropped"])
@pytest.mark.parametrize("tension", list(TensionComputationMethod))
def test_every_tension_method_gives_the_librarys_numbers(reconstructions, name, tension):
    child, library = reconstructions[name]
    mesh = child.value.mesh()
    for pressure in PressureComputationMethod:
        outcome, _, _ = infer_on_mesh(mesh, _options(tension, pressure))
        try:
            expected = quietly(infer_forces, library, mode_tension=tension, mode_pressure=pressure)
        except Exception as refused:  # the library refuses: the plugin reports it and keeps what it can
            assert outcome.status in ("failed", "pressures refused"), refused
            assert outcome.error
            if outcome.tensions is not None:
                assert same_floats(outcome.tensions, quietly(infer_tensions, library, mode=tension))
            if name == "cropped" and not pressure.has_crop_face_formulation:
                assert "no formulation for crop faces" in outcome.error
            continue
        assert outcome.status == "complete"
        assert same_floats(outcome.tensions, expected[0])
        assert same_floats(outcome.pressures, expected[1])


def test_cut_cells_pressures_are_nan_and_warned(reconstructions):
    child, _ = reconstructions["cropped"]
    mesh = child.value.mesh()
    outcome, warnings_summary, _ = infer_on_mesh(
        mesh, _options(TensionComputationMethod.YoungDupre, PressureComputationMethod.Variational)
    )
    cut = [int(c) for c in mesh.cut_cells]
    assert cut
    assert all(outcome.pressures[c] != outcome.pressures[c] for c in cut)  # nan
    assert "CutCellPressureWarning" in {w["category"] for w in warnings_summary}
