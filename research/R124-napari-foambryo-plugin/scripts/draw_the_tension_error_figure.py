"""Draw the relative tension errors of benchmark case 000: the true mask through the plugin, and a membrane
segmentation plugin's labels as it leaves them (A) and with its background pieces merged (B).

Reads the plugin's own export of case 000 (`tensions.csv`, the true mask) and the manual check's record
(`from_a_membrane_segmentation_plugin_on_case_000.json`, variants A and B); the true tensions are the plugin's
sample data. Scale matched as the benchmark does (mean of true / mean of inferred).

Usage: python draw_the_tension_error_figure.py EXPORTED_CASE_000_FOLDER CHECK_JSON FIGURE_PNG
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from napari_foambryo._sample_data import true_values


def main(exported: Path, check_json: Path, figure_png: Path) -> None:
    truth, _ = true_values()
    with (exported / "tensions.csv").open() as handle:
        inferred = {(int(r["label_1"]), int(r["label_2"])): float(r["tension"]) for r in csv.DictReader(handle)}
    shared = [k for k in truth if k in inferred]
    factor = np.mean([truth[k] for k in shared]) / np.mean([inferred[k] for k in shared])
    series = {"true mask\n(the plugin's sample)": [abs(inferred[k] * factor - truth[k]) / truth[k] for k in shared]}
    record = json.loads(check_json.read_text())
    labels = {"A_as_the_plugin_leaves_it": "membrane plugin, A:\nbackground in 6 pieces",
              "B_background_pieces_merged": "membrane plugin, B:\npieces merged (fill tool)"}
    for name, label in labels.items():
        series[label] = list(record["variants"][name]["relative_error_per_interface"].values())
    figure = Figure(figsize=(7.2, 3.6), dpi=200)
    axes = figure.add_subplot(1, 1, 1)
    rng = np.random.default_rng(0)
    for x, (label, errors) in enumerate(series.items()):
        jitter = rng.uniform(-0.12, 0.12, len(errors))
        axes.scatter(np.full(len(errors), x) + jitter, errors, s=14, color="#1f5f99", alpha=0.7, zorder=3)
        median = float(np.median(errors))
        axes.plot([x - 0.25, x + 0.25], [median, median], color="#c0392b", lw=2, zorder=4)
        axes.text(x + 0.28, median, f"median {median:.3f}\n{len(errors)} interfaces", va="center", fontsize=7)
    axes.set_xticks(range(len(series)))
    axes.set_xticklabels(list(series), fontsize=7)
    axes.set_xlim(-0.5, len(series) - 0.1)
    axes.set_ylabel("relative tension error\n(scale-matched)", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.set_title("Benchmark case 000 (6 cells): tensions inferred in napari against the true ones", fontsize=8,
                   loc="left")
    figure.tight_layout()
    figure.savefig(figure_png)
    print(f"wrote {figure_png}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
