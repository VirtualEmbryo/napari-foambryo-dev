# R134 — the maintainer's feedback on the two viewers (pointer)

The record of this item is in foambryo, on `research/R134-viewer-feedback-napari-and-polyscope`
(`research/R134-viewer-feedback-napari-and-polyscope/`: README, REPORT, HANDOFF, decisions, scripts, results, figures).

What it changed here, on `feature/first-version` (`c59151c`, on `7613194`):

* a **View** dock: **explode** (each cell moved by the library's scattered-view displacement, bitwise
  `compute_scattered_arrays`; each interface drawn on both of its cells' sides) and **cut** (a plane normal to z, y
  or x, napari's clipping planes on every layer of the run and on its Labels layer);
* a **pressures** check box in Infer: off, the tensions only, as `foambryo infer --no-pressures`;
* a **Diagnostics** tab in Results (one row per linear solve; the singular values on a log scale), and Export writes
  `solves.csv` beside `spectra.npz`; Export says when the Infer dock's options differ from the inference it exports;
* Export **asks** (Replace or Cancel) before replacing foambryo's files in a folder; no other file is touched;
* the mask report is information, not a warning.

Plugin suite: 106 passed, 0 skipped (tiers T1-T3, gates G1-G11 unchanged). The specification's amendments are its
§ 12. The morning kit is refreshed: `scratch/napari-foambryo/try_napari_foambryo.zsh` and `TRY-IT.md` (walk 6).
