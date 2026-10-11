"""The manual check of the segmentation-plugin route, driven in a real napari window, with screenshots.

A user segments benchmark case 000's synthetic membrane stack with another napari plugin and goes on to foambryo
without saving a file. The segmentation plugin is napari-segment-blobs-and-things-with-membranes (local-minima
seeded watershed, `spot_sigma=8`, `outline_sigma=0`, chosen by `tune_a_membrane_segmentation_of_case_000.py`),
installed in the scratch venv only. Its Labels layer has no voxel size and labels the background 1, in six pieces.

* Variant A, as the plugin leaves it: voxel size from the image layer (voxel units confirmed), *medium is label 1*.
  The five other background pieces become cells that touch the image box (the mask report says so before Run).
* Variant B, after the edit a user makes with napari's fill tool: the background pieces that touch the image box
  set to 1 (simulated here by assigning the label; the rule uses no ground truth: the true cells do not touch the
  box). The results are marked out of date until Reconstruct runs again.

Each segment is matched to a true cell of `000_labels_filled.tif` by its largest overlap (IoU > 0.5; the medium to
0), and the inferred tensions are compared with case 000's true tensions after matching the scale (the benchmark's
convention: mean of true / mean of inferred over the compared interfaces). Writes a JSON of results and
screenshots.

Usage (the scratch venv, from scratch/napari-foambryo/):
    python manual_check_from_a_membrane_segmentation_plugin.py DATASET_DIR RESULTS_JSON SCREENSHOT_DIR
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

SPOT_SIGMA, OUTLINE_SIGMA = 8, 0
LABELS_NAME = "local_minima_seeded_watershed result"


def match_segments(segments: np.ndarray, truth: np.ndarray, medium: int) -> dict[int, int]:
    """Segment label -> true label, by the largest overlap with IoU > 0.5; the medium label -> 0."""
    mapping = {medium: 0}
    for label in np.unique(segments):
        if label == medium:
            continue
        inside = segments == label
        values, counts = np.unique(truth[inside], return_counts=True)
        best = int(values[np.argmax(counts)])
        if best > 0 and counts.max() / np.count_nonzero(inside | (truth == best)) > 0.5:
            mapping[int(label)] = best
    return mapping


def compare(tensions: dict, mapping: dict[int, int], truth_tensions: dict) -> dict:
    """Relative errors after matching the scale, on the interfaces whose two labels map to true labels."""
    mapped = {}
    unmatched = 0
    for (a, b), t in tensions.items():
        if a in mapping and b in mapping and np.isfinite(t):
            key = tuple(sorted((mapping[a], mapping[b])))
            if key in truth_tensions:
                mapped[key] = float(t)
                continue
        unmatched += 1
    factor = np.mean([truth_tensions[k] for k in mapped]) / np.mean(list(mapped.values()))
    errors = {f"{k[0]}-{k[1]}": abs(t * factor - truth_tensions[k]) / truth_tensions[k] for k, t in mapped.items()}
    return {"interfaces_compared": len(mapped), "true_interfaces": len(truth_tensions),
            "interfaces_not_compared": unmatched, "scale_factor": float(factor),
            "median_relative_error": float(np.median(list(errors.values()))),
            "relative_error_per_interface": {k: round(float(v), 4) for k, v in sorted(errors.items())}}


def main(dataset: Path, results_json: Path, screenshots: Path) -> None:
    import napari
    from napari_segment_blobs_and_things_with_membranes import local_minima_seeded_watershed
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from napari_foambryo._layers import RUNS
    from napari_foambryo._sample_data import true_values
    from napari_foambryo._widgets import ResultsWidget

    screenshots.mkdir(parents=True, exist_ok=True)
    truth = tifffile.imread(dataset / "000_labels_filled.tif")
    truth_tensions, _ = true_values()
    viewer = napari.Viewer(title="napari-foambryo — from a segmentation plugin")
    viewer.window.resize(1700, 1050)
    image = viewer.open(str(dataset / "000_microscopy.tif"), name="000_microscopy")[0]
    started = time.perf_counter()
    segments = np.asarray(local_minima_seeded_watershed(image.data, spot_sigma=SPOT_SIGMA,
                                                        outline_sigma=OUTLINE_SIGMA))
    segmentation_seconds = time.perf_counter() - started
    labels = viewer.add_labels(segments, name=LABELS_NAME)  # as the plugin's widget adds it: no voxel size
    docks = {}
    for name in ("Reconstruct", "Infer", "Export"):
        _, docks[name] = viewer.window.add_plugin_dock_widget("napari-foambryo", name)
    docks["Results"] = ResultsWidget(viewer)
    viewer.window.add_dock_widget(docks["Results"], name="Results", area="right")
    viewer.dims.ndisplay = 3
    reconstruct, infer = docks["Reconstruct"], docks["Infer"]
    record = {"segmentation_plugin": "napari-segment-blobs-and-things-with-membranes",
              "segmentation_call": f"local_minima_seeded_watershed(image, spot_sigma={SPOT_SIGMA}, "
                                   f"outline_sigma={OUTLINE_SIGMA})",
              "segmentation_seconds": round(segmentation_seconds, 2), "segments": int(len(np.unique(segments))),
              "background_label": int(segments[0, 0, 0]), "variants": {}}
    steps = iter(["setup", "wait_report", "reconstruct", "wait_reconstruct", "infer", "wait_infer", "edit",
                  "wait_report", "reconstruct", "wait_reconstruct", "infer", "wait_infer", "export", "done"])
    state = {"step": next(steps), "variant": "A_as_the_plugin_leaves_it"}

    def shot(name: str) -> None:
        QApplication.processEvents()
        viewer.screenshot(str(screenshots / f"{name}.png"), canvas_only=False)

    def tick() -> None:
        step = state["step"]
        variant = state["variant"]
        if step == "setup":
            reconstruct.select_layer(LABELS_NAME)
            reconstruct.spacing_from.setCurrentText("layer: 000_microscopy")
            reconstruct.confirm_voxels.setChecked(True)
            reconstruct.medium_box.setValue(record["background_label"])
        elif step == "wait_report":
            text = reconstruct.report_label.text()
            if "Mask report (nothing" not in text or text == state.get("previous_report"):
                QTimer.singleShot(500, tick)
                return
            record["variants"].setdefault(variant, {})["mask_report_text"] = reconstruct.report_label.text()
            shot(f"07_{variant}_mask_report")
        elif step == "reconstruct":
            reconstruct.start()
        elif step == "wait_reconstruct":
            if reconstruct._worker is not None:
                QTimer.singleShot(500, tick)
                return
            record["variants"][variant]["reconstruct"] = reconstruct.status.text()
        elif step == "infer":
            infer.start()
        elif step == "wait_infer":
            if infer._worker is not None:
                QTimer.singleShot(500, tick)
                return
            run = RUNS[infer.session.current]
            mapping = match_segments(run.mask, truth, 0)
            result = compare(run.outcome.tensions, mapping, truth_tensions)
            result.update(status=run.outcome.status, cut_cells=[int(c) for c in run.mesh.cut_cells],
                          infer=infer.status.text(), results_summary=docks["Results"].summary.text(),
                          mapping={str(k): v for k, v in mapping.items()})
            record["variants"][variant].update(result)
            viewer.reset_view()
            shot(f"08_{variant}_tensions")
        elif step == "edit":
            box_pieces = [int(v) for v in np.unique(np.concatenate([np.unique(f) for f in (
                segments[0], segments[-1], segments[:, 0], segments[:, -1], segments[:, :, 0], segments[:, :, -1])]))
                if v != record["background_label"]]
            edited = segments.copy()
            edited[np.isin(edited, box_pieces)] = record["background_label"]
            labels.data = edited  # what the fill tool does, piece by piece
            record["edit"] = {"pieces_set_to_the_background": box_pieces,
                              "rule": "the pieces that touch the image box (the mask report lists them)"}
            record["out_of_date_after_edit"] = viewer.layers[f"{LABELS_NAME} (medium = 0) cells [out of date]"].name
            shot("09_out_of_date_after_the_edit")
            state["variant"] = "B_background_pieces_merged"
            state["previous_report"] = record["variants"]["A_as_the_plugin_leaves_it"]["mask_report_text"]
        elif step == "export":
            docks["Export"].folder.setText(str(screenshots.parent / "exports" / "from_a_segmentation_plugin"))
            docks["Export"].overwrite.setChecked(True)
            record["exported"] = docks["Export"].export()
        elif step == "done":
            results_json.write_text(json.dumps(record, indent=1) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "variants"}, indent=1), flush=True)
            for name, variant_record in record["variants"].items():
                print(name, variant_record.get("median_relative_error"), variant_record.get("interfaces_compared"),
                      variant_record.get("interfaces_not_compared"), flush=True)
            viewer.close()
            return
        state["step"] = next(steps)
        QTimer.singleShot(300, tick)

    QTimer.singleShot(1500, tick)
    napari.run()


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
