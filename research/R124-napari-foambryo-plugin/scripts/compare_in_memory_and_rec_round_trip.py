"""Do the Python route and the command's route give bitwise the same tensions and pressures for one mask?

Python route (what a napari plugin calling the library in memory would compute):
    dcel_mesh_from_segmentation_mask(mask, spacing=spacing) -> infer_forces(mesh)
Command route (`foambryo mesh` then `foambryo infer`):
    construct_mesh_in_physical_units -> save_to_rec_mesh (crop faces written as the last triangle rows)
    -> dcel_mesh_from_file -> infer_forces

The two meshes hold the same triangles; the .rec file only reorders them when the mesh has crop faces. Whether
that reordering changes the inferred numbers is the question. Synthetic masks only (four Voronoi cells in a ball,
whole or cut by the image box), the fixture of foambryo's command-line tests. Prints JSON; writes only to a
temporary directory under this script's folder, removed at the end.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from dw3d import get_default_mesh_reconstruction_algorithm
from foambryo import dcel_mesh_from_file, dcel_mesh_from_segmentation_mask, infer_forces
from foambryo.pressure_inference import PressureComputationMethod
from foambryo.tension_inference import TensionComputationMethod


def four_cells_in_a_ball(n: int = 36, centre_z: float | None = None) -> np.ndarray:
    z, y, x = np.indices((n, n, n))
    c = (n - 1) / 2
    cz = c if centre_z is None else centre_z
    ball = (z - cz) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= 14**2
    seeds = np.array([[cz - 6, c - 5, c], [cz + 6, c - 5, c + 1], [cz, c + 7, c - 6], [cz + 1, c + 4, c + 8]])
    distances = ((np.stack([z, y, x], axis=-1)[..., None, :] - seeds) ** 2).sum(axis=-1)
    return np.where(ball, distances.argmin(axis=-1) + 1, 0).astype(np.uint16)


def bitwise_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() for k in a)


def max_relative_difference(a: dict, b: dict) -> float:
    worst = 0.0
    for k in a:
        x, y = float(a[k]), float(b[k])
        if np.isfinite(x) and np.isfinite(y) and x != y:
            worst = max(worst, abs(x - y) / max(abs(x), abs(y)))
    return worst


def main() -> None:
    masks = {"whole": four_cells_in_a_ball(), "cut_by_the_image_box": four_cells_in_a_ball(centre_z=6.0)}
    spacings = {"cubic": (1.0, 1.0, 1.0), "anisotropic_z2": (2.0, 1.0, 1.0)}
    methods = [
        (TensionComputationMethod.YoungDupre, PressureComputationMethod.Variational),
        (TensionComputationMethod.ForceBalance, PressureComputationMethod.Variational),
        (TensionComputationMethod.Lami, PressureComputationMethod.Variational),
    ]
    work = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
    out = []
    try:
        for mask_name, mask in masks.items():
            for spacing_name, spacing in spacings.items():
                in_memory = dcel_mesh_from_segmentation_mask(mask, spacing=spacing)
                algorithm = get_default_mesh_reconstruction_algorithm(3, False)
                algorithm.construct_mesh_in_physical_units(mask, spacing, spacing_unit="um", axis_order="zyx")
                rec = work / f"{mask_name}_{spacing_name}.rec"
                algorithm.save_to_rec_mesh(rec)
                round_trip = dcel_mesh_from_file(rec)
                for tension_mode, pressure_mode in methods:
                    t1, p1 = infer_forces(in_memory, mode_tension=tension_mode, mode_pressure=pressure_mode)
                    t2, p2 = infer_forces(round_trip, mode_tension=tension_mode, mode_pressure=pressure_mode)
                    out.append({
                        "mask": mask_name, "spacing": spacing_name,
                        "crop_faces": int(np.count_nonzero(in_memory.crop_faces)),
                        "tension_method": tension_mode.name, "pressure_method": pressure_mode.name,
                        "same_points": bool(np.array_equal(in_memory.v, round_trip.v)),
                        "tensions_bitwise_equal": bitwise_equal(t1, t2),
                        "tensions_max_relative_difference": max_relative_difference(t1, t2),
                        "pressures_bitwise_equal": bitwise_equal(p1, p2),
                        "pressures_max_relative_difference": max_relative_difference(p1, p2),
                    })
    finally:
        shutil.rmtree(work)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
