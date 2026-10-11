"""Choose the parameters of a membrane segmentation plugin on benchmark case 000's synthetic microscopy stack.

The plugin is napari-segment-blobs-and-things-with-membranes (local-minima seeded watershed), installed in the
scratch venv only; it is not a dependency of napari-foambryo. For each (spot_sigma, outline_sigma), the number of
segments, the background's label (the label of the image corner), and each true cell's best intersection over
union with a segment (the true mask is `000_labels_filled.tif`). Prints one JSON line per setting.

Usage: python tune_a_membrane_segmentation_of_case_000.py DATASET_DIR
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from napari_segment_blobs_and_things_with_membranes import local_minima_seeded_watershed


def best_iou(truth: np.ndarray, segments: np.ndarray) -> dict[int, float]:
    out = {}
    for cell in np.unique(truth[truth > 0]):
        inside = truth == cell
        labels, counts = np.unique(segments[inside], return_counts=True)
        best = labels[np.argmax(counts)]
        union = np.count_nonzero(inside | (segments == best))
        out[int(cell)] = round(float(counts.max() / union), 3)
    return out


def main(dataset: Path) -> None:
    image = tifffile.imread(dataset / "000_microscopy.tif")
    truth = tifffile.imread(dataset / "000_labels_filled.tif")
    for spot_sigma in (4, 6, 8, 10):
        for outline_sigma in (0, 1, 2):
            started = time.perf_counter()
            segments = np.asarray(local_minima_seeded_watershed(image, spot_sigma=spot_sigma,
                                                                outline_sigma=outline_sigma))
            print(json.dumps({"spot_sigma": spot_sigma, "outline_sigma": outline_sigma,
                              "segments": int(len(np.unique(segments))), "dtype": str(segments.dtype),
                              "background_label": int(segments[0, 0, 0]),
                              "best_iou_per_true_cell": best_iou(truth, segments),
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
