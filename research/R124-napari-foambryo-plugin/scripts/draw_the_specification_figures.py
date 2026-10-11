"""Draw the two figures of the specification.

1. `plugin_widgets_layers_and_code_path.png`: what the plugin shows (widgets, layers, results dock), what runs
   where (the GUI thread, a worker thread, a child process), and which library functions it calls, shared with
   the `foambryo` command and with Python.
2. `estimate_and_display_cost.png`: the measured reconstruction costs the pre-run estimate is fitted to, with
   the fit and its stated error band; and the size of the display copies against the reconstruction's peak
   memory, on the 1 866-cell tissue's mesh.

Usage: python draw_the_specification_figures.py RESULTS_DIR FIGURES_DIR
RESULTS_DIR holds `reconstruction_cost_fit.json` and `display_copy_sizes.json` (written by the scripts beside this
one). Needs matplotlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#222222"
MUTED = "#666666"
GUI = "#dbe9f6"
WORKER = "#fde9c9"
CHILD = "#f9d5cf"
LIBRARY = "#d9f0d3"
FILES = "#ececec"


def box(ax, x, y, w, h, text, face, *, size=8, weight="normal", align="center"):  # noqa: ANN001, ANN201
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0015,rounding_size=0.006", linewidth=0.8,
                           edgecolor=INK, facecolor=face)
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.008
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color=INK, weight=weight, linespacing=1.25)
    return patch


def arrow(ax, start, end, *, text=None, colour=INK, style="-|>", dashed=False, offset=(0.0, 0.012)):  # noqa: ANN001, ANN201
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, linewidth=0.9, color=colour,
                                 linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(mx, my, text, fontsize=6.5, color=colour, ha="center", va="bottom")


def draw_code_path(path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 8.2), dpi=170)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.012, 0.975, "napari-foambryo 0.1.0 (proposed): widgets, layers and the one code path", fontsize=12,
            weight="bold", color=INK, va="top")
    ax.text(0.012, 0.947, "Colours: blue = napari GUI thread; orange = worker thread; red = child process "
            "(terminated on Cancel); green = public library functions, shared with the foambryo command and "
            "Python; grey = files.", fontsize=7.5, color=MUTED, va="top")

    # --- napari window -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.012, 0.06), 0.43, 0.855, boxstyle="round,pad=0.004", linewidth=1.0,
                                edgecolor=INK, facecolor="white"))
    ax.text(0.022, 0.908, "napari window", fontsize=9, weight="bold", va="top")

    ax.text(0.022, 0.884, "Dock widgets", fontsize=8, weight="bold", va="top", color=MUTED)
    box(ax, 0.022, 0.735, 0.2, 0.12,
        "Reconstruct\nLabels layer ▾   spacing z, y, x\n(from the layer's scale)   unit\n"
        "▸ Advanced: min_distance\nestimate: time, memory   [Run] [Cancel]", GUI, size=6.6, align="left")
    box(ax, 0.022, 0.585, 0.2, 0.135,
        "Infer\nmesh ▾   tension method   pressure method\n"
        "○ one mesh   ○ seed ensemble (N seeds)\n"
        "▸ Advanced: mean tension, base pressure,\n   solver, diagnostics, combine, angle\n"
        "   reading (ensemble), workers   [Run] [Cancel]", GUI, size=6.6, align="left")
    box(ax, 0.022, 0.475, 0.2, 0.095,
        "Export\nfolder   ☐ include the mask (.npz)\n[Export]  mesh.rec, tensions.csv,\n"
        "pressures.csv, run.json, colour_bars.png",
        GUI, size=6.6, align="left")
    box(ax, 0.235, 0.475, 0.198, 0.38,
        "Results dock\n\nInterfaces | Cells\n─────────────────\nlabel 1  label 2  tension\n"
        "  [seed SD  rel. SD  seeds found  flag]\n  undetermined (nan)\n\n"
        "label  pressure  cut cell  undetermined\n─────────────────\nselect a row → highlight\n"
        "click a cell in Labels → select row\n[Export CSV]  (exact floats)\n\ncolour bars: save_colour_bars,\n"
        "the scale of foambryo view\n\nwarnings relayed, once each", GUI, size=6.6, align="left")

    ax.text(0.022, 0.466, "Layers (all in the spacing's unit, z, y, x)", fontsize=8, weight="bold", va="top",
            color=MUTED)
    layers = [
        ("Labels (user's): scale = voxel size, translate t", "voxel centre j sits at  j·scale + t"),
        ("<mask> cells: Surface, one copy per cell, Labels colours", "mesh vertex i·h + t: same frame"),
        ("<mask> tensions: Surface, vertices copied per interface", "1 866 cells: 263 337 vertices, 19.5 MiB"),
        ("<mask> pressures: Surface, copied per higher-labelled cell", "1 866 cells: 194 939 vertices, 16.4 MiB"),
        ("<mask> crop faces: Surface, grey, translucent", "only when the labels reach the image box"),
        ("<mask> highlight: the selected interface or cell", "recoloured copy, no new geometry"),
    ]
    for i, (name, note) in enumerate(layers):
        y = 0.4 - i * 0.056
        box(ax, 0.022, y, 0.24, 0.042, name, GUI if i else "#eef4fb", size=6.5, align="left")
        ax.text(0.27, y + 0.022, note, fontsize=6.3, color=MUTED, va="center")

    # --- worker / child -------------------------------------------------------------------------
    box(ax, 0.48, 0.78, 0.19, 0.105,
        "Worker thread (napari thread_worker)\nprogress at step level, elapsed vs\nestimate; polls the child; Cancel\n"
        "terminates the child's process group", WORKER, size=6.6)
    box(ax, 0.48, 0.6, 0.19, 0.14,
        "Child process (spawn)\nruns one library call:\nreconstruct the mask, or the seed\n"
        "ensemble (its own workers inside);\n"
        "returns the mesh by pickling\n(bitwise the in-process mesh, measured)", CHILD, size=6.6)
    box(ax, 0.48, 0.47, 0.19, 0.095,
        "In the worker thread (seconds):\ntensions then pressures,\n"
        "kept separately when the pressure\nsolve refuses",
        WORKER, size=6.6)
    box(ax, 0.48, 0.32, 0.19, 0.12,
        "Estimate before Run\nt ≈ 3.6e-8 s · V^1.16,  M ≈ 2.1e-5 GiB · V^0.70\n"
        "V = voxels given to dw3d (× anisotropy)\nshown with its band (×/÷ 1.6 time, 1.5 memory)\n"
        "against the machine's memory;\nshown in Reconstruct and Infer", WORKER, size=6.3)

    arrow(ax, (0.433, 0.83), (0.48, 0.83), text="")
    ax.text(0.437, 0.838, "Run\n(Reconstruct,\nensemble)", fontsize=5.8, color=INK, va="bottom")
    arrow(ax, (0.433, 0.515), (0.48, 0.515), text="")
    ax.text(0.437, 0.523, "Run\n(Infer)", fontsize=5.8, color=INK, va="bottom")
    arrow(ax, (0.575, 0.78), (0.575, 0.74), text="")
    arrow(ax, (0.48, 0.66), (0.433, 0.66), text="mesh", colour=MUTED, offset=(0, 0.004))
    arrow(ax, (0.48, 0.49), (0.433, 0.49), text="results", colour=MUTED, offset=(0, -0.022))

    # --- library --------------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.7, 0.3), 0.288, 0.615, boxstyle="round,pad=0.004", linewidth=1.0,
                                edgecolor=INK, facecolor="white"))
    ax.text(0.71, 0.9, "Public library API\n(foambryo ≥ 0.7, delaunay-watershed-3d ≥ 0.7)", fontsize=8.2,
            weight="bold", va="top")
    library = [
        ("reconstruct a mask in physical units\n(proposed public: the command's mesh step)",
         "dw3d construct_mesh_in_physical_units\nresample to cubic voxels if anisotropic; crop faces"),
        ("infer the tensions, then the pressures\n(proposed public: the command's infer step)",
         "infer_tensions, infer_pressures\nas infer_forces calls them"),
        ("infer_tensions_over_seeds", "seed ensembles; no single mesh, no pressures"),
        ("result tables and run record\n(proposed public: today private to the command)",
         "tensions.csv, pressures.csv, run.json schema 1\nexact floats"),
        ("colour scales and colour bars\n(proposed public: today private)",
         "tension_colours, pressure_colours, the one\npressure scale, save_colour_bars"),
        ("the mask checks and readers", "3-D non-negative integer labels; .npy .npz .tif"),
    ]
    for i, (name, note) in enumerate(library):
        y = 0.78 - i * 0.078
        box(ax, 0.71, y, 0.268, 0.068, "", LIBRARY)
        ax.text(0.716, y + 0.047, name, fontsize=6.5, va="center", color=INK, weight="bold", linespacing=1.15)
        ax.text(0.716, y + 0.015, note, fontsize=6.0, va="center", color=MUTED, linespacing=1.1)
    arrow(ax, (0.67, 0.69), (0.71, 0.81), text="")
    arrow(ax, (0.67, 0.65), (0.71, 0.659), text="")
    arrow(ax, (0.67, 0.52), (0.71, 0.737), text="")
    arrow(ax, (0.445, 0.24), (0.48, 0.24), text="", colour=MUTED)
    ax.text(0.447, 0.25, "Export", fontsize=5.8, color=MUTED)

    # --- the other front ends -------------------------------------------------------------------
    box(ax, 0.7, 0.175, 0.135, 0.09, "foambryo command\nmesh · infer ·\ninfer-over-seeds · view", FILES, size=6.6)
    box(ax, 0.853, 0.175, 0.135, 0.09, "Python\ndcel_mesh_from_segmentation_mask\n+ infer_forces", FILES, size=6.6)
    arrow(ax, (0.767, 0.265), (0.767, 0.3), style="-|>")
    arrow(ax, (0.92, 0.265), (0.92, 0.3), style="-|>")
    box(ax, 0.48, 0.175, 0.19, 0.09, "Export folder = the command's files\nmesh.rec byte for byte foambryo mesh's;\n"
        "tables bitwise; run.json schema 1\n+ the equivalent command lines", FILES, size=6.4)
    ax.text(0.48, 0.14, "Same mask, same options, same library versions → the same 64-bit numbers\nin napari, on the "
            "command line and in Python. Measured on a four-cell fixture (whole and\ncut by the image box; cubic "
            "and z-step 2× voxels; 3 tension methods): the in-memory\nroute and the .rec route agree bitwise, 12/12.",
            fontsize=7, color=INK, va="top")
    ax.text(0.022, 0.035, "Sizes: the 1 866-cell tissue's mesh by the cubic-voxel route (134 538 vertices, 326 805 "
            "triangles, 11 979 interfaces; delaunay-watershed-3d 5bc1132, foambryo 2da5186). Arrays as handed to "
            "napari (float64 vertices and values, float32 RGBA, int64 faces).", fontsize=6.5, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def draw_estimate_and_display_cost(results: Path, path: Path) -> None:
    fit = json.loads((results / "reconstruction_cost_fit.json").read_text())
    sizes = json.loads((results / "display_copy_sizes.json").read_text())
    anchors = {  # the anchor points of fit_reconstruction_cost_against_voxels.py, in the same order
        "benchmark at real cell size (median)": (1.5e6, 0.8, 0.6),
        "benchmark default size (median)": (7.72e6, 2.3, 1.2),
        "90-cell simulated tissue": (4.2e7, 21.0, 4.7),
        "110-cell tissue, native grid": (7.99e7, 35.0, 5.1),
        "165-cell simulated tissue": (1.17e8, 76.0, 11.6),
        "1 866-cell tissue, native grid": (4.98e8, 610.0, 29.1),
        "1 866-cell tissue, cubic voxels": (8.3e8, 807.0, 45.6),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), dpi=170, gridspec_kw={"width_ratios": [1, 1, 1.05]})
    v = np.logspace(6, 9.3, 100)
    for ax, key, column, unit, title in (
        (axes[0], "seconds", 1, "s", "Reconstruction wall time"),
        (axes[1], "peak_GiB", 2, "GiB", "Reconstruction peak memory (RSS)"),
    ):
        a, b = fit[key]["prefactor"], fit[key]["exponent"]
        band = fit[key]["max_factor_error"]
        ax.fill_between(v, a * v**b / band, a * v**b * band, color="#cfe3f3", label=f"band ×/÷ {band:g}")
        ax.plot(v, a * v**b, color="#1f5f99", lw=1.2, label=f"fit  {a:.2g} · V^{b:.3g}")
        for name, values in anchors.items():
            ax.scatter(values[0], values[column], s=16, color=INK, zorder=3)
            # The 90- and 110-cell anchors sit close together: put the 90-cell label above and to the left.
            above_left = name.startswith("90-cell")
            ax.annotate(name, (values[0], values[column]), fontsize=5.6, color=MUTED,
                        xytext=(-4, 4) if above_left else (3, -8), ha="right" if above_left else "left",
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("voxels given to dw3d, V (after resampling to cubic voxels)", fontsize=7.5)
        ax.set_ylabel(unit, fontsize=7.5)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    axes[1].axhline(128, color="#b03a2e", lw=0.8, ls="--")
    axes[1].set_ylim(top=400)
    axes[1].text(1.2e6, 135, "this machine: 128 GiB", fontsize=6.3, color="#b03a2e")

    tissue = sizes["a_1866_cell_tissue_cubic_voxel_route"]
    labels = ["plain mesh", "tensions copy\n(per interface)", "pressures copy\n(per higher label)",
              "cells copy\n(per cell side)", "fully unindexed\n(3 per triangle)"]
    values = [tissue["plain_mesh_MiB"], tissue["tension_copy_MiB"], tissue["pressure_surface_copy_MiB"],
              tissue["pressure_copy_MiB"], tissue["unindexed_copy_MiB"]]
    ax = axes[2]
    ax.barh(range(len(values)), values, color=["#999999", "#1f5f99", "#7b3294", "#5aae61", "#bbbbbb"])
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:g} MiB", va="center", fontsize=7)
    ax.set_yticks(range(len(values)), labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 70)
    ax.set_xlabel("MiB of arrays handed to napari", fontsize=7.5)
    ax.set_title("Display copies, 1 866-cell tissue's mesh\n(reconstruction peak: 36-50 GiB)", fontsize=8.5,
                 loc="left")
    ax.tick_params(labelsize=7)
    fig.text(0.01, 0.005, "Anchors: per-reconstruction medians from the research records (default reconstruction, "
             "min_distance 3, relocation on, Apple M3 Max, 128 GiB; timings mostly on a shared machine, provisional; "
             "peak RSS reliable). The band is the largest in-sample ratio, not a validated prediction interval.",
             fontsize=6.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    results_dir, figures_dir = Path(sys.argv[1]), Path(sys.argv[2])
    figures_dir.mkdir(parents=True, exist_ok=True)
    draw_code_path(figures_dir / "plugin_widgets_layers_and_code_path.png")
    draw_estimate_and_display_cost(results_dir, figures_dir / "estimate_and_display_cost.png")
