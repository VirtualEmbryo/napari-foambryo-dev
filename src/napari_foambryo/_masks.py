"""From whatever a Labels layer holds to the mask the library reconstructs, with every accommodation recorded.

A Labels layer may come from a file, from napari's own tools or from another segmentation plugin. What the plugin
accepts, and how, is decided here, without Qt, so that it can be tested alone:

* the full-resolution level of a multiscale layer;
* a lazy array (dask, zarr) read into memory;
* a leading axis of length 1 (time or channel) dropped, with a note;
* any integer type, labels not consecutive (a label is a name, and it is kept);
* the medium label swapped with 0 when the user says the medium is another label.

Refused, with the reason: several time points, 2-D data, floating-point or boolean data, negative labels (the
library's own checks, `foambryo.segmentation_masks.check_segmentation_mask`). Nothing else is changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from foambryo.segmentation_masks import SegmentationMaskError, check_segmentation_mask

__all__ = ["PreparedMask", "array_sha256", "prepare_mask", "swap_medium_label"]


@dataclass(frozen=True)
class PreparedMask:
    """The mask to reconstruct, and what was done to get it from the layer's data.

    Attributes:
        mask: the 3-D integer mask, axes (z, y, x).
        notes: one sentence per accommodation, shown to the user and written into `run.json`.
        dropped_leading_axis: whether a leading axis of length 1 was dropped.
        multiscale_level: the level read from a multiscale layer (0, the full resolution), else `None`.
    """

    mask: np.ndarray
    notes: tuple[str, ...] = field(default_factory=tuple)
    dropped_leading_axis: bool = False
    multiscale_level: int | None = None


def prepare_mask(data: Any, *, multiscale: bool = False, name: str = "the layer") -> PreparedMask:  # noqa: ANN401
    """Turn a Labels layer's data into a mask, or say why it cannot be one.

    Args:
        data: the layer's data (an array, or the list of levels of a multiscale layer).
        multiscale: whether `data` is a list of levels, finest first.
        name: how messages name the layer.

    Returns:
        PreparedMask: the mask and the notes.

    Raises:
        SegmentationMaskError: for several time points, 2-D data, non-integer data or negative labels.
    """
    notes: list[str] = []
    level = None
    if multiscale:
        level = 0
        data = data[0]
        notes.append(f"{name} is multiscale: its full-resolution level is used")
    lazy = not isinstance(data, np.ndarray)
    array = np.asarray(data)
    if lazy:
        notes.append(f"{name} holds a lazy array: it was read into memory ({array.nbytes / 2**20:.1f} MiB)")
    dropped = False
    if array.ndim == 4:
        if array.shape[0] != 1:
            message = (
                f"{name} holds {array.shape[0]} time points or channels (shape {array.shape}); "
                "one mask is one 3-D frame: extract the frame first"
            )
            raise SegmentationMaskError(message)
        array = array[0]
        dropped = True
        notes.append(f"{name} has a leading axis of length 1: it was dropped, shape {tuple(array.shape)}")
    mask = check_segmentation_mask(array, name=name)
    return PreparedMask(mask=mask, notes=tuple(notes), dropped_leading_axis=dropped, multiscale_level=level)


def swap_medium_label(mask: np.ndarray, medium_label: int) -> np.ndarray:
    """A copy of the mask with `medium_label` and 0 swapped: the library's medium is label 0.

    Args:
        mask: the mask.
        medium_label: the label the segmentation gave the medium; 0 returns the mask itself, unchanged.

    Returns:
        np.ndarray: the mask, or a swapped copy of the same dtype.
    """
    if medium_label == 0:
        return mask
    if medium_label < 0 or medium_label > np.iinfo(mask.dtype).max:
        message = f"the medium label must be a label of the mask's type ({mask.dtype}), not {medium_label}"
        raise SegmentationMaskError(message)
    swapped = mask.copy()
    swapped[mask == medium_label] = 0
    swapped[mask == 0] = medium_label
    return swapped


def array_sha256(array: np.ndarray) -> str:
    """The SHA-256 of an array's C-order bytes, as hexadecimal (with its dtype and shape it identifies the mask)."""
    digest = hashlib.sha256()
    flat = np.ascontiguousarray(array).reshape(-1)
    step = max(1, (1 << 24) // max(1, flat.itemsize))
    for start in range(0, flat.size, step):
        digest.update(flat[start : start + step].tobytes())
    return digest.hexdigest()
