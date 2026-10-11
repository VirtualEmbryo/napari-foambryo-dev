"""Build napari-foambryo's sample data file for benchmark case 000, from the benchmarking dataset.

Case 000 of the benchmark of the method's 2023 paper (Ichbiah et al., Nature Methods, doi:10.1038/s41592-023-02084-7):
a simulated 6-cell aggregate at mechanical equilibrium, with its true tensions and pressures. Writes one compressed
`.npz` with plain arrays only (no pickled objects):

* `mask` (uint8, 194 x 200 x 199, axes z, y, x; 0 the medium), from `000_labels_filled.tif`;
* `spacing` (1, 1, 1) and `spacing_unit` "voxel": the simulation is in voxel units;
* `true_tension_label_1`, `true_tension_label_2`, `true_tension` (one row per interface), from
  `000_dict_tensions.npy`;
* `true_pressure_label`, `true_pressure`, from `000_dict_pressures.npy`.

Usage: python make_case_000_sample_data.py DATASET_DIR OUTPUT.npz
"""

import sys
from pathlib import Path

import numpy as np
import tifffile


def main(dataset_dir: Path, output: Path) -> None:
    mask = tifffile.imread(dataset_dir / "000_labels_filled.tif")
    assert mask.min() >= 0 and mask.max() < 256, "labels must fit in uint8"
    tensions = np.load(dataset_dir / "000_dict_tensions.npy", allow_pickle=True).item()
    pressures = np.load(dataset_dir / "000_dict_pressures.npy", allow_pickle=True).item()
    keys = sorted((int(a), int(b)) for a, b in tensions)
    labels = sorted(int(k) for k in pressures)
    np.savez_compressed(
        output,
        mask=mask.astype(np.uint8),
        spacing=np.array([1.0, 1.0, 1.0]),
        spacing_unit=np.array("voxel"),
        true_tension_label_1=np.array([k[0] for k in keys], dtype=np.int64),
        true_tension_label_2=np.array([k[1] for k in keys], dtype=np.int64),
        true_tension=np.array([float(tensions[k]) for k in keys]),
        true_pressure_label=np.array(labels, dtype=np.int64),
        true_pressure=np.array([float(pressures[k]) for k in labels]),
    )
    print(f"wrote {output}: {output.stat().st_size} bytes; {len(keys)} interfaces, {len(labels)} materials")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
