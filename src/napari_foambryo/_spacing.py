"""Which voxel size a reconstruction uses: from the Labels layer, from another layer, or typed; never guessed.

* *This layer*: the layer's `scale` (napari's (z, y, x) order, dw3d's), with its unit when napari records one, else
  `um` (the command's fallback). A sample's metadata states its voxel size and unit. napari's own default scale
  (1, 1, 1) without a unit means *unknown*: Run waits for an explicit confirmation of voxel units (decision Q3).
* *Another layer* (typically the image that was segmented): the same, read from that layer, which must have the
  labels' 3-D shape.
* *Typed*: three numbers > 0 and a unit.

The spacing is then checked as the library checks it (`dw3d.mask_preprocessing.coarse_axis_and_factor`: one axis
may be longer than the other two, which must be equal), and an anisotropic one is announced with its cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from foambryo.segmentation_masks import SPACING_UNIT_PATTERN

__all__ = ["SpacingChoice", "layer_spacing", "resolve_spacing"]

THIS_LAYER = "this layer"
TYPED = "typed"
VOXEL_UNIT = "voxel"


@dataclass(frozen=True)
class SpacingChoice:
    """The spacing to use, or why there is none yet.

    Attributes:
        spacing: `(dz, dy, dx)`.
        unit: its unit.
        spacing_from: what `run.json` records: `"this layer"`, `"layer '<name>'"`, `"typed"`, with how it was known.
        problem: why Run cannot start (`None` when it can).
        needs_confirmation: the voxel size is unknown and waits for *voxel units* to be confirmed.
        message: what the widget shows about the spacing.
    """

    spacing: tuple[float, float, float]
    unit: str
    spacing_from: str
    problem: str | None = None
    needs_confirmation: bool = False
    message: str = ""


def _layer_shape(layer: Any) -> tuple[int, ...]:  # noqa: ANN401
    data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
    return tuple(int(n) for n in data.shape)


def layer_spacing(layer: Any) -> tuple[tuple[float, float, float], str | None, str | None]:  # noqa: ANN401
    """A layer's voxel size, its unit (`None` if napari records none) and how it is known (`None`: unknown)."""
    scale = tuple(float(v) for v in layer.scale[-3:])
    info = getattr(layer, "metadata", {}).get("foambryo", {})
    if "spacing" in info and tuple(float(v) for v in info["spacing"]) == scale:
        return scale, str(info.get("spacing_unit", "um")), "the sample's metadata"
    units = {str(u) for u in tuple(getattr(layer, "units", ()))[-3:]}
    unit = units.pop() if len(units) == 1 else None
    unit = None if unit in (None, "pixel") else unit
    if unit is None and all(v == 1.0 for v in scale):
        return scale, None, None
    return scale, unit, "the layer's scale" + ("" if unit else " (no unit recorded: um assumed, as the command does)")


def resolve_spacing(
    labels: Any,  # noqa: ANN401 - the Labels layer
    mode: str,
    *,
    mask_shape: tuple[int, ...],
    other: Any = None,  # noqa: ANN401 - another layer, for mode "layer"
    typed: tuple[float, float, float] | None = None,
    typed_unit: str = "um",
    voxel_units_confirmed: bool = False,
) -> SpacingChoice:
    """Decide the spacing for a reconstruction (see the module's docstring)."""
    if mode == TYPED:
        if typed is None or len(typed) != 3 or not all(math.isfinite(v) and v > 0 for v in typed):
            return SpacingChoice((1.0, 1.0, 1.0), typed_unit, TYPED, "type three voxel sizes greater than 0")
        if not SPACING_UNIT_PATTERN.fullmatch(typed_unit or ""):
            return SpacingChoice(tuple(typed), typed_unit, TYPED, "the unit must be one ASCII word (um, nm, ...)")
        return _checked(SpacingChoice(tuple(float(v) for v in typed), typed_unit, TYPED))
    source = labels if mode == THIS_LAYER else other
    if source is None:
        return SpacingChoice((1.0, 1.0, 1.0), "um", mode, "choose the layer to take the voxel size from")
    if source is not labels and _layer_shape(source)[-3:] != tuple(mask_shape):
        return SpacingChoice(
            (1.0, 1.0, 1.0),
            "um",
            f"layer '{source.name}'",
            f"'{source.name}' has the shape {_layer_shape(source)[-3:]}, the labels "
            f"{tuple(mask_shape)}: its voxel size cannot be the labels'",
        )
    spacing, unit, how = layer_spacing(source)
    label = THIS_LAYER if source is labels else f"layer '{source.name}'"
    if how is None:
        if not voxel_units_confirmed:
            return SpacingChoice(
                spacing,
                VOXEL_UNIT,
                label,
                "voxel size unknown (napari's default scale): enter it, "
                "take it from the image layer, or confirm voxel units",
                needs_confirmation=True,
            )
        return _checked(SpacingChoice(spacing, VOXEL_UNIT, f"{label} (voxel units confirmed)"))
    return _checked(SpacingChoice(spacing, unit or "um", f"{label} ({how})"))


def _checked(choice: SpacingChoice) -> SpacingChoice:
    from dw3d.mask_preprocessing import coarse_axis_and_factor, isotropic_spacing

    try:
        found = coarse_axis_and_factor(choice.spacing)
    except ValueError as error:
        return SpacingChoice(choice.spacing, choice.unit, choice.spacing_from, str(error))
    spacing_text = " × ".join(f"{v:g}" for v in choice.spacing)
    if found is None:
        message = f"voxels {spacing_text} {choice.unit} (cubic), from {choice.spacing_from}"
    else:
        axis, factor = found
        message = (
            f"voxels {spacing_text} {choice.unit}, from {choice.spacing_from}: {factor:.3g} times longer along "
            f"{'zyx'[axis]}. Resampled to cubic voxels of {isotropic_spacing(choice.spacing):g} {choice.unit} "
            f"before meshing: {factor:.3g}× the voxels; min_distance counts cubic voxels"
        )
    return SpacingChoice(choice.spacing, choice.unit, choice.spacing_from, None, False, message)
