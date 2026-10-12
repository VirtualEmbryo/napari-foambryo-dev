"""The estimate of a reconstruction's time and memory before it runs, and what the widget does with it.

The estimate is foambryo's own (`foambryo.reconstruction_cost`), the one every front end gives: power laws of the
number of voxels handed to delaunay-watershed-3d (the mask's voxels times the anisotropy, after resampling to cubic
voxels), fitted to measured reconstructions. Each estimate is a central value and a band: the largest in-sample ratio
between an anchor and the fit, **not** a validated prediction interval. Time and memory also depend on the number of
cells and the tissue's shape, and the anchors come from one machine. No constant is kept here.

The memory guard compares the band with the machine's memory (decision Q8): within what is free, the estimate is
shown plainly; above what is free but with the central value within the machine, an amber warning; when the
central value exceeds the machine's memory, Run needs *run anyway* to be ticked.
"""

from __future__ import annotations

from dataclasses import dataclass

from foambryo.reconstruction_cost import (
    ReconstructionCostEstimate,
    estimate_reconstruction_cost,
    voxels_handed_to_dw3d,
)

__all__ = ["CostEstimate", "MemoryVerdict", "estimate_reconstruction", "memory_verdict", "voxels_handed_to_dw3d"]

GIB = 2**30

#: A reconstruction's estimated wall time and peak memory, each `(low, central, high)`: foambryo's.
CostEstimate = ReconstructionCostEstimate


def estimate_reconstruction(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> CostEstimate:
    """The estimate for a mask of this shape and voxel size (`foambryo.reconstruction_cost`)."""
    return estimate_reconstruction_cost(voxels_handed_to_dw3d(shape, spacing))


def estimate_text(estimate: CostEstimate) -> str:
    """For example *"about 13 min (8–20 min) and 39 GiB (26–59 GiB)"*."""
    return estimate.text(dash="–")


@dataclass(frozen=True)
class MemoryVerdict:
    """What the widget shows next to the estimate, and whether Run needs *run anyway*.

    Attributes:
        level: `"ok"`, `"amber"` or `"red"`.
        message: the sentence shown.
        needs_run_anyway: whether Run is enabled only once *run anyway* is ticked.
    """

    level: str
    message: str
    needs_run_anyway: bool


def memory_verdict(estimate: CostEstimate, total_bytes: int, available_bytes: int) -> MemoryVerdict:
    """Compare the estimate with the machine's memory (decision Q8)."""
    total, available = total_bytes / GIB, available_bytes / GIB
    _, central, high = estimate.peak_gib
    if high <= available:
        return MemoryVerdict("ok", f"this machine has {total:.0f} GiB, {available:.0f} GiB free", False)
    if central <= total:
        return MemoryVerdict(
            "amber",
            f"may need more memory than is free ({available:.0f} GiB of {total:.0f} GiB): close other programs",
            False,
        )
    return MemoryVerdict(
        "red",
        f"needs about {central:.0f} GiB; this machine has {total:.0f} GiB. Run `foambryo mesh` on a larger machine "
        "(same numbers) and open the .rec here, or tick 'run anyway'",
        True,
    )
