# Phase B — the overnight run's decision log (11–12 October 2026)

Unattended run: nobody answers questions, so each choice below is the most conservative one that keeps the
specification's principles (one code path, no default of its own, nothing silent), with its reason.

## Milestones

| | state | commits |
|---|---|---|
| M0 set-up | done | worktrees `scratch/worktrees/public-helpers-for-front-ends/`, `napari-foambryo-first-version/`, `napari-foambryo-dw3d-master/` (detached at dw3d 68eccf4); scratch venv `scratch/napari-foambryo/venv` |
| M1 foambryo public helpers | built; suites below | foambryo `feature/public-helpers-for-front-ends`: ac8cd0c, 6daeebe, ce2125a, c274e54 |
| M2 specification version 1 | done | `research/R124-napari-foambryo-plugin` 7ba4a41 |
| M1 checks | done | foambryo suite 927 passed + 1 skipped on 669a976, 951 passed + 1 skipped on c274e54 (the 24 new tests; same skip); ruff 23 = 23; ty 189 → 179 |
| M3 a usable plugin | done | `napari-foambryo` `feature/first-version`: adcb9f4 (.DS_Store), 806f2ee (the plugin), 9769ecb (fixes from the real window); plugin suite 75 passed, 0 skipped (T1–T3; G1–G3, G6–G8, G10; G9 in its first form) |
| M4 ensembles, estimate, cancel | done | `feature/first-version` 74e6d11; plugin suite 83 passed, 0 skipped; gates G4 (child = library, 1 and 2 workers), G5 (in-memory = .rec route on case 000 cut by the image box), G9 (anchors within the band; the guard's three levels; `run anyway` in the widget); Cancel stops a slow child and its own process |
| M5 from another plugin's segmentation | done | `feature/first-version` a09cdd8, 15f52ab, 63e4a39; plugin suite 86 passed, 0 skipped; gate G11 (bitwise through the widgets); manual check with napari-segment-blobs-and-things-with-membranes 0.3.12 (BSD-3-Clause, R. Haase) on case 000's microscopy stack, in a real window: median relative tension error 0.074 as the plugin leaves the labels, 0.069 once the background pieces are merged, against 0.041 on the true mask (`results/from_a_membrane_segmentation_plugin_on_case_000.json`) |
| M3b the morning kit | done | `scratch/napari-foambryo/try_napari_foambryo.zsh`, `start_napari_with_foambryo.py`, `TRY-IT.md`; launcher run once with `--smoke-test` in a real napari 0.9.2 window: case 000 reconstructed (1712 points, 10.8 s, 0.76 GiB), inferred, exported; screenshots in `scratch/napari-foambryo/screenshots/` |

## Decisions

1. **Worktree names.** The overnight file asks for `scratch/worktrees/napari-foambryo-phase-b/`; M0 and M1 were
   built before it arrived, in worktrees named after their work. Kept: moving them would break the scratch venv's
   editable installs (absolute paths) in the middle of the run. Names are in the table above.
2. **Public modules are façades.** `foambryo.results` and `foambryo.colour_scales` re-export the private
   `_result_tables` and `_colour_scales` instead of renaming them, so every old import path keeps working
   (R133's branch rebases cleanly) and the helpers branch rebases cleanly after the command's review.
3. **`foambryo mesh` prints the mask report in its own commit (c274e54).** Hervé asked both for "the command's
   tests unchanged" (Q1) and for `foambryo mesh` to print the mask report (segmentation-plugin section, point 5);
   the two conflict on the one test that pins mesh's standard output. The three helper commits pass the command's
   tests unchanged; the fourth adds the line and changes that one assertion, so it can be reviewed or dropped
   alone. The line is printed only once the spacing is accepted, so a refused command line still prints nothing
   on standard output (two other tests pin that).
4. **Mask report definitions.** Pieces are connected through shared voxel faces (6 neighbours), as a mesh sees
   them; "very small" is below 5 % of the median cell's voxels (`DEFAULT_SMALL_LABEL_FRACTION`), stated in the
   printed line. Counting is slab by slab, so the report never copies a large mask whole.
5. **The reconstruction result carries the `.rec` bytes.** `reconstruct_mask` writes dw3d's own export into a
   temporary directory and keeps its bytes, so the plugin's `mesh.rec` is `foambryo mesh`'s byte for byte without
   pickling dw3d's algorithm object (which holds the tessellation) from the child process.
6. **ty**: run with delaunay-watershed-3d's `ty` against foambryo's venv: 189 diagnostics on 669a976, 179 on the
   helpers tip; the only lines not in the baseline are three that moved verbatim from `cli.py` into
   `results.py` with the warning collector. Ruff: the same 23 findings as 669a976.
7. **napari's layer right-click menu is not open to plugins** (`napari/layers/context` is not in napari 0.9.2's
   contributable menus, measured on the installed version). *Reconstruct with foambryo* is under **Layers ›
   Measure** and acts on the selected Labels layer; the Reconstruct widget's own layer list does the same.
8. **Version floors kept at 0.7.** The plugin requires `delaunay-watershed-3d>=0.7` and `foambryo[cli]>=0.7`, as
   specified; both sources read 0.6.0, so the plugin is installed in the scratch venv with `--no-deps` (its
   dependencies are installed there explicitly). `matplotlib` is a declared dependency.
9. **Sample data.** Case 000 ships as a 85 kB `.npz` of plain arrays (no pickle), with its voxel size in the
   layer's metadata (voxel units: the simulation's), so that it needs no confirmation of the voxel size. The
   generated foam is 12 Voronoi cells on a 96³ grid (radius 44): at the specified 48³ the reconstruction had 111
   points, too coarse to show anything; at 96³ it has about 630 points and reconstructs in about 2 s. The foams
   have a nominal 0.5 um voxel.
10. **Unknown voxel size confirmed as voxel units** records the unit `voxel` (not the command's `um` fallback),
    which says what the numbers are.
11. **No status-bar value on mesh layers.** napari's Surface layer shows its `vertex_values` under the cursor;
    the plugin colours by `vertex_colors` and passes no values (nan values would upset napari's contrast limits).
    The tables are the reading.
12. **Single-mesh inference runs in a worker thread, also with `solver="dense"`** in 0.1.0 (the specification
    asks for a child process with the dense solver so that it can be cancelled): left for M4.
13. **What napari shows in 3-D.** The labels' voxels hide the mesh, so Reconstruct hides the source Labels layer
    in 3-D and says so; Infer shows the tensions and hides the cells and pressures (same geometry) and says so;
    selecting a row shows its layer. Nothing about the data changes.
14. **A real window found what headless tests did not**: napari hands each plugin widget its own
    `PublicOnlyProxy` of the viewer, so the widgets' shared session must be keyed by the viewer behind the proxy
    (fixed in 9769ecb; the widget tests now give each widget its own proxy). Also: the spawn-based runner hangs
    when Python reads its main script from standard input; the launcher is a script file.
15. **Scale matching for the known answer**: the Results dock's relative error uses the benchmark's convention,
    `mean(true) / mean(inferred)` over the shared interfaces (`foambryo_benchmarks.metrics`).
16. **Ensemble on screen** (decision Q4): `<mask> ensemble tensions (N seeds, shown on seed 42's mesh)`, the
    ensemble's tensions on the reconstruction's own mesh; its colour range is that of the ensemble's tensions.
    The Cells tab is empty for an ensemble (no single mesh carries pressures). An ensemble needs a mesh
    reconstructed in the plugin (it reconstructs the mask again); a mesh opened from a `.rec` file is refused
    with the reason.
17. **The ensemble export's `input`** names `mask.npz` and the array's SHA-256 (there is no mask file unless the
    mask is exported); its tables are byte for byte `foambryo infer-over-seeds`'.
18. **Suggested workers** = free memory / the per-seed estimate's upper end, at least 1; the widget turns amber
    when the chosen number exceeds it and never lowers the user's choice.
19. **The segmentation plugin for the manual check** is napari-segment-blobs-and-things-with-membranes 0.3.12
    (BSD-3-Clause, by Robert Haase; local-minima seeded watershed for membrane images): classic image processing,
    no model file, no GPU; installed in the scratch venv only (`pip freeze` before and after in
    `scratch/napari-foambryo/logs/`: nothing already installed changed). Parameters chosen on the true mask
    (`tune_a_membrane_segmentation_of_case_000.py`): `spot_sigma=8`, `outline_sigma=0`, the smallest that finds
    all 6 cells (IoU 0.975–0.982).
20. **The plugin labels the background 1, in six pieces.** Variant A uses it as it comes (*medium is label 1*): the
    five other pieces become cells that touch the image box (cut cells; the mask report says so before Run,
    "11 cells; 5 touch the image box"); the 18 true interfaces are all found (median error 0.074), with 23 extra
    interfaces with the fake cells, and the reconstruction's peak memory is 7.7 GiB against 1.4 GiB. Variant B
    sets the pieces that touch the image box to 1, the edit a user makes with napari's fill tool (simulated by
    assigning the label; the rule uses no ground truth): 6 cells, median error 0.069. Matching segments to true
    cells uses the true mask (largest overlap, IoU > 0.5) only for the comparison.
21. **Two defects found by the manual check, fixed and tested**: the mask report did not follow a change of the
    labels (it does now, painting included); with a swapped medium, editing the original segmentation did not
    mark the run out of date (the run watches both layers now). Also found by G11 and fixed: the voxel size from
    another layer (napari's `LayerList` has no `get`, which a lint suggestion had introduced), and the swapped copy
    of a layer with a leading axis.
22. **A worker's result arriving after its dock was closed** touched deleted Qt widgets (it aborted the
    interpreter at the end of one test session); handlers now do nothing for a closed dock. The one commit made
    on an unchecked log (a09cdd8: `tail` succeeded where pytest's exit code was not read) was re-run: 84 passed;
    later commits are gated on pytest's own exit code.
