"""Tier T1: what is accepted as a mask, which spacing is used, and the estimate before a run.

The estimate's gate (G9) is here in its first form: the fit reproduces its own anchors within its stated band, and
the memory guard's three levels fall where decision Q8 puts them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from foambryo.segmentation_masks import SegmentationMaskError

from napari_foambryo._estimate import GIB, estimate_reconstruction, estimate_text, memory_verdict, voxels_handed_to_dw3d
from napari_foambryo._masks import array_sha256, prepare_mask, swap_medium_label
from napari_foambryo._spacing import THIS_LAYER, TYPED, resolve_spacing


def test_a_leading_axis_of_length_1_is_dropped_and_said():
    prepared = prepare_mask(np.zeros((1, 4, 5, 6), dtype=np.uint16), name="'seg'")
    assert prepared.mask.shape == (4, 5, 6) and prepared.dropped_leading_axis
    assert "leading axis of length 1" in prepared.notes[0]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((3, 4, 5, 6), dtype=np.uint16), "3 time points"),
        (np.zeros((4, 5), dtype=np.uint16), "2-D"),
        (np.zeros((4, 5, 6)), "float64"),
        (np.zeros((4, 5, 6), dtype=bool), "bool"),
        (-np.ones((4, 5, 6), dtype=np.int16), "negative"),
    ],
)
def test_what_is_refused_says_why(data, message):
    with pytest.raises(SegmentationMaskError, match=message):
        prepare_mask(data)


def test_multiscale_reads_the_full_resolution_and_lazy_arrays_are_read():
    dask = pytest.importorskip("dask.array")
    full = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    prepared = prepare_mask([dask.from_array(full), full[::2, ::2, ::2]], multiscale=True)
    assert np.array_equal(prepared.mask, full) and prepared.multiscale_level == 0
    assert any("lazy" in note for note in prepared.notes)


def test_the_medium_swap_swaps_two_labels_and_nothing_else():
    mask = np.array([[[1, 1, 2], [0, 3, 1]]], dtype=np.uint16)
    swapped = swap_medium_label(mask, 1)
    assert swapped.tolist() == [[[0, 0, 2], [1, 3, 0]]]
    assert swapped.dtype == mask.dtype
    assert swap_medium_label(mask, 0) is mask
    assert array_sha256(mask) == array_sha256(mask.copy())


class _Layer:
    def __init__(self, shape, scale, units=("pixel",) * 3, metadata=None, name="layer"):
        self.data = np.zeros(shape, dtype=np.uint8)
        self.scale, self.units, self.metadata, self.name, self.multiscale = scale, units, metadata or {}, name, False


def test_the_spacing_is_never_guessed():
    unknown = _Layer((4, 5, 6), (1.0, 1.0, 1.0))
    choice = resolve_spacing(unknown, THIS_LAYER, mask_shape=(4, 5, 6))
    assert choice.problem and choice.needs_confirmation
    confirmed = resolve_spacing(unknown, THIS_LAYER, mask_shape=(4, 5, 6), voxel_units_confirmed=True)
    assert confirmed.problem is None and confirmed.unit == "voxel"
    image = _Layer((4, 5, 6), (2.0, 0.5, 0.5), units=("micrometer",) * 3, name="membranes")
    from_image = resolve_spacing(unknown, "layer", mask_shape=(4, 5, 6), other=image)
    assert from_image.spacing == (2.0, 0.5, 0.5) and from_image.unit == "micrometer"
    assert from_image.spacing_from.startswith("layer 'membranes'")
    assert "4 times longer along z" in from_image.message
    wrong_shape = resolve_spacing(unknown, "layer", mask_shape=(4, 5, 6), other=_Layer((4, 5, 7), (2.0, 1.0, 1.0)))
    assert "cannot be the labels'" in wrong_shape.problem
    assert "two different" in resolve_spacing(unknown, TYPED, mask_shape=(4, 5, 6), typed=(2.0, 1.0, 0.5)).problem


def test_the_estimate_is_the_librarys_and_reproduces_its_anchors_within_its_band():
    """The plugin keeps no copy of the fit: its estimate is `foambryo.reconstruction_cost`'s, bitwise."""
    from foambryo.reconstruction_cost import RECONSTRUCTION_COST_ANCHORS, estimate_reconstruction_cost

    for name, voxels, seconds, peak_gib in RECONSTRUCTION_COST_ANCHORS:
        side = voxels ** (1 / 3)
        estimate = estimate_reconstruction((side, side, side), (1.0, 1.0, 1.0))
        assert estimate.seconds[0] * 0.999 <= seconds <= estimate.seconds[2] * 1.001, name
        assert estimate.peak_gib[0] * 0.999 <= peak_gib <= estimate.peak_gib[2] * 1.001, name
    assert voxels_handed_to_dw3d((10, 10, 10), (2.0, 1.0, 1.0)) == 2000.0
    assert estimate_reconstruction((100, 200, 200), (2.0, 1.0, 1.0)) == estimate_reconstruction_cost(8e6)
    import napari_foambryo

    assert not list(Path(napari_foambryo.__file__).parent.glob("estimate/*.json"))


def test_the_memory_guard_needs_run_anyway_only_beyond_the_machine():
    estimate = estimate_reconstruction((1000, 1000, 830), (1.0, 1.0, 1.0))  # 8.3e8 voxels: about 39 GiB
    assert memory_verdict(estimate, 128 * GIB, 100 * GIB).level == "ok"
    assert memory_verdict(estimate, 128 * GIB, 30 * GIB).level == "amber"
    red = memory_verdict(estimate, 32 * GIB, 30 * GIB)
    assert red.level == "red" and red.needs_run_anyway and "foambryo mesh" in red.message
    assert "min" in estimate_text(estimate)
    assert "GiB" in estimate_text(estimate)


def test_a_tiny_estimate_does_not_read_zero_seconds():
    assert "0 s" not in estimate_text(estimate_reconstruction((10, 10, 10), (1.0, 1.0, 1.0)))


@pytest.mark.parametrize("shape", [(10, 10, 10), (100, 100, 100), (110, 110, 110)], ids=["1e3", "1e6", "1.3e6"])
def test_a_reconstruction_under_a_second_reads_under_1_s_once(shape):
    text = estimate_text(estimate_reconstruction(shape, (1.0, 1.0, 1.0)))
    assert text.startswith("under 1 s and "), text
    assert "about under" not in text


def test_a_seed_ensemble_is_worded_like_one_reconstruction():
    from napari_foambryo._estimate import ensemble_estimate  # new in 0.7.0: imported here so older code collects

    per_seed = estimate_reconstruction((100, 100, 100), (1.0, 1.0, 1.0))
    ensemble = ensemble_estimate(per_seed, rounds=3, workers=2)
    assert ensemble.seconds == tuple(3 * s for s in per_seed.seconds)
    assert ensemble.peak_gib == tuple(2 * g for g in per_seed.peak_gib)
    assert estimate_text(ensemble_estimate(estimate_reconstruction((10, 10, 10), (1.0, 1.0, 1.0)), 1, 1)) == (
        "under 1 s and 3 MiB (2 MiB–4 MiB)"
    )
    large = ensemble_estimate(estimate_reconstruction((350, 350, 350), (1.0, 1.0, 1.0)), rounds=4, workers=2)
    assert estimate_text(large).startswith("about ") and " min" in estimate_text(large)
