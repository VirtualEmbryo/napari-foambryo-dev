"""Manual smoke runs in a real napari window: the cropped sample, then a real-size 90-cell simulated tissue.

Drives the docks as a user clicks them (Reconstruct, Infer), takes a screenshot after each, and writes what the
docks said. On the 90-cell tissue (a simulated tissue at real cell size with known tensions, voxel units), the
inferred tensions are compared with the true ones restricted to the reference mesh's interfaces, after matching
the scale (mean of true / mean of inferred). Run under the workspace's heavy-run lock (the real-size step).

Usage (the scratch venv, from scratch/napari-foambryo/):
    python smoke_run_on_a_cropped_foam_and_a_90_cell_tissue.py MASK_NPZ TRUE_TENSIONS_NPY RESULTS_JSON SCREENSHOT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(mask_npz: Path, truth_npy: Path, results_json: Path, screenshots: Path) -> None:
    import napari
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from napari_foambryo._layers import RUNS
    from napari_foambryo._widgets import ResultsWidget

    screenshots.mkdir(parents=True, exist_ok=True)
    with np.load(mask_npz, allow_pickle=False) as data:
        tissue = data["mask"]
    truth = {(int(a), int(b)): float(t) for (a, b), t in np.load(truth_npy, allow_pickle=True).item().items()}
    viewer = napari.Viewer(title="napari-foambryo — smoke runs")
    viewer.window.resize(1700, 1050)
    docks = {}
    for name in ("Reconstruct", "Infer"):
        _, docks[name] = viewer.window.add_plugin_dock_widget("napari-foambryo", name)
    docks["Results"] = ResultsWidget(viewer)
    viewer.window.add_dock_widget(docks["Results"], name="Results", area="right")
    viewer.dims.ndisplay = 3
    viewer.open_sample("napari-foambryo", "foam_cropped")
    metadata = {"foambryo": {"spacing": (1.0, 1.0, 1.0), "spacing_unit": "voxel",
                             "true_tensions": [[a, b, t] for (a, b), t in truth.items()]}}
    viewer.add_labels(tissue, name="90-cell simulated tissue", metadata=metadata)
    record: dict = {}
    jobs = iter([("generated foam, cut by the image box", "10_cropped_foam"),
                 ("90-cell simulated tissue", "11_90_cell_tissue")])
    state = {"job": next(jobs), "step": "reconstruct"}
    reconstruct, infer = docks["Reconstruct"], docks["Infer"]

    def shot(name: str) -> None:
        QApplication.processEvents()
        viewer.screenshot(str(screenshots / f"{name}.png"), canvas_only=False)

    def tick() -> None:
        layer_name, prefix = state["job"]
        step = state["step"]
        if step == "reconstruct":
            for layer in viewer.layers:
                layer.visible = layer.name.startswith(layer_name)
            reconstruct.select_layer(layer_name)
            record[layer_name] = {"estimate": reconstruct.estimate_label.text(),
                                  "spacing": reconstruct.spacing_label.text()}
            reconstruct.start()
            state["step"] = "wait_reconstruct"
        elif step == "wait_reconstruct":
            if reconstruct._worker is None:
                record[layer_name]["reconstruct"] = reconstruct.status.text()
                record[layer_name]["mask_report"] = reconstruct.report_label.text()
                infer.start()
                state["step"] = "wait_infer"
        elif step == "wait_infer":
            if infer._worker is None:
                run = RUNS[infer.session.current]
                record[layer_name].update(infer=infer.status.text(), results=docks["Results"].summary.text(),
                                          status=run.outcome.status, cut_cells=len(run.mesh.cut_cells),
                                          undetermined_pressures=int(sum(not np.isfinite(p) for p in
                                                                         run.outcome.pressures.values())))
                viewer.reset_view()
                shot(f"{prefix}_tensions")
                try:
                    state["job"] = next(jobs)
                    state["step"] = "reconstruct"
                except StopIteration:
                    results_json.write_text(json.dumps(record, indent=1) + "\n")
                    for name, entry in record.items():
                        print(name, "|", entry["reconstruct"][:90], "|", entry["results"][:200], flush=True)
                    viewer.close()
                    return
        QTimer.singleShot(500, tick)

    QTimer.singleShot(1500, tick)
    napari.run()


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
