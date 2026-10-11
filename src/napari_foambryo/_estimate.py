"""The estimate of a reconstruction's time and memory before it runs, and what the widget does with it.

Power laws of the number of voxels handed to delaunay-watershed-3d (the mask's voxels times the anisotropy, after
resampling to cubic voxels), fitted to measured per-reconstruction medians (`estimate/reconstruction_cost.json`).
Each estimate is a central value and a band: the largest in-sample ratio between an anchor and the fit, **not** a
validated prediction interval. Time and memory also depend on the number of cells and the tissue's shape, and the
anchors come from one machine.

The memory guard compares the band with the machine's memory (decision Q8): within what is free, the estimate is
shown plainly; above what is free but with the central value within the machine, an amber warning; when the
central value exceeds the machine's memory, Run needs *run anyway* to be ticked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

__all__ = ["CostEstimate", "MemoryVerdict", "estimate_reconstruction", "memory_verdict", "voxels_handed_to_dw3d"]

GIB = 2**30


@cache
def cost_model() -> dict:
    """The anchors and fitted constants, from the package's data file."""
    return json.loads(files("napari_foambryo").joinpath("estimate/reconstruction_cost.json").read_text())


def voxels_handed_to_dw3d(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> float:
    """The voxels the reconstruction meets: the mask's, times the anisotropy when it is resampled to cubic voxels."""
    voxels = float(shape[0]) * float(shape[1]) * float(shape[2])
    return voxels * max(spacing) / min(spacing)


@dataclass(frozen=True)
class CostEstimate:
    """A reconstruction's estimated wall time and peak memory, each with its band.

    Attributes:
        voxels: the voxels handed to delaunay-watershed-3d.
        seconds: `(low, central, high)`.
        peak_gib: `(low, central, high)`.
    """

    voxels: float
    seconds: tuple[float, float, float]
    peak_gib: tuple[float, float, float]

    def text(self) -> str:
        """For example *"about 13 min (8–20 min) and 39 GiB (26–59 GiB)"*."""
        return (
            f"about {_duration(self.seconds[1])} ({_duration(self.seconds[0])}–{_duration(self.seconds[2])}) and "
            f"{_memory(self.peak_gib[1])} ({_memory(self.peak_gib[0])}–{_memory(self.peak_gib[2])})"
        )


def estimate_reconstruction(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> CostEstimate:
    """The estimate for a mask of this shape and voxel size."""
    voxels = voxels_handed_to_dw3d(shape, spacing)
    out = {}
    for key in ("seconds", "peak_gib"):
        fit = cost_model()["fit"][key]
        central = fit["prefactor"] * voxels ** fit["exponent"]
        out[key] = (central / fit["band_factor"], central, central * fit["band_factor"])
    return CostEstimate(voxels, out["seconds"], out["peak_gib"])


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


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "under 1 s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _memory(gib: float) -> str:
    return f"{gib * 1024:.0f} MiB" if gib < 1 else f"{gib:.0f} GiB"
