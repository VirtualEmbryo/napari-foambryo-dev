"""Shared fixtures: headless Qt, napari's settings kept out of the user's folder, the samples reconstructed once.

The tests are of three kinds, named in their docstrings: T1 needs no Qt; T2 uses `napari.components.ViewerModel` (no
canvas); T3 drives the widgets with pytest-qt on a `ViewerModel` under Qt's offscreen platform. A test that needs a
real viewer window (an OpenGL context) is marked `opengl` and skipped unless `NAPARI_FOAMBRYO_OPENGL=1`.
"""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY.parent / "benchmarking-dataset"


@pytest.fixture(autouse=True, scope="session")
def _napari_settings_in_a_temporary_folder(tmp_path_factory: pytest.TempPathFactory) -> None:
    os.environ.setdefault("NAPARI_CONFIG", str(tmp_path_factory.mktemp("napari") / "settings.yaml"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("NAPARI_FOAMBRYO_OPENGL") == "1":
        return
    skip = pytest.mark.skip(reason="needs an OpenGL context: set NAPARI_FOAMBRYO_OPENGL=1 with a display")
    for item in items:
        if "opengl" in item.keywords:
            item.add_marker(skip)


def sha(array: np.ndarray) -> str:
    """The SHA-256 of an array's bytes (sums or counts miss changes)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def same_floats(a: dict, b: dict) -> bool:
    """The same keys and, per key, bitwise the same 64-bit float (`nan` equal to `nan`)."""
    if set(a) != set(b):
        return False
    for key in a:
        x, y = float(a[key]), float(b[key])
        if math.isnan(x) or math.isnan(y):
            if not (math.isnan(x) and math.isnan(y)):
                return False
        elif np.float64(x).tobytes() != np.float64(y).tobytes():
            return False
    return True


def quietly(function, *args, **kwargs):
    """Call a library function with its warnings silenced (the tests compare numbers, not messages)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="session")
def samples() -> dict[str, tuple[np.ndarray, tuple[float, float, float]]]:
    """The generated foams, by name: (mask, spacing)."""
    from napari_foambryo._sample_data import foam_anisotropic, foam_cropped, foam_cubic

    out = {}
    for name, make in (("cubic", foam_cubic), ("anisotropic", foam_anisotropic), ("cropped", foam_cropped)):
        ((mask, meta, _),) = make()
        out[name] = (mask, tuple(meta["scale"]))
    return out
