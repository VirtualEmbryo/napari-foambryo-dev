# R124 — napari-foambryo: the specification (phase A)

**Fragment.** Phase A of two: the specification only, written for the maintainer's review. No plugin code exists
yet; phase B (the build) starts only after written approval of `docs/SPECIFICATION.md`.

## Question

What should a napari plugin over delaunay-watershed-3d and foambryo do, and how can it give exactly the numbers of
the library and of the `foambryo` command? Four facts the specification needed were measured rather than assumed.

## What was measured

| question | script | result | file |
|---|---|---|---|
| Do the in-memory route (`dcel_mesh_from_segmentation_mask` + `infer_forces`) and the command's `.rec` route (crop faces written as the last rows) give the same numbers? | `scripts/compare_in_memory_and_rec_round_trip.py` | bitwise equal, 12 of 12 (four-cell fixture whole and cut by the image box; cubic and z-step-2 voxels; YoungDupre, ForceBalance, Lami with Variational pressures). On the cut fixture, WeightedLaplace pressures are refused by the library (`ValueError`, no crop-face formulation). | `results/in_memory_vs_rec_round_trip.json` |
| Does a mesh built in a spawned child process give bitwise the in-process numbers? | `scripts/check_mesh_survives_a_child_process.py` | yes (points, triangles, tensions, pressures); 0.8 s round trip including the spawn; pickled mesh 120 kB on the fixture | `results/mesh_survives_a_child_process.json` |
| How large are the display copies napari's per-vertex colouring needs? | `scripts/count_display_copy_sizes.py` | on the 1 866-cell tissue's mesh (cubic-voxel route): 19.5 MiB per-interface copy, 16.4 MiB per-higher-label copy, 26.2 MiB per-cell-side copy, against 13.6 MiB for the plain mesh | `results/display_copy_sizes.json` |
| Can the pre-run estimate be a simple function of the voxels handed to dw3d? | `scripts/fit_reconstruction_cost_against_voxels.py` | time ≈ 3.6e-8 s · V^1.157 (in-sample within ×/÷ 1.58), peak ≈ 2.1e-5 GiB · V^0.702 (×/÷ 1.49), seven anchors quoted from earlier research records | `results/reconstruction_cost_fit.json` |

Figures (`scripts/draw_the_specification_figures.py`, written to `docs/figures/`): the widgets, layers and code
path; the estimate's anchors and fit beside the display-copy sizes.

## Provenance

* foambryo `feature/the-foambryo-command` at 669a976 (on the Python path), delaunay-watershed-3d `master` at
  68eccf4 (editable in its repository venv); Python 3.12.0, NumPy 2.5.1, SciPy 1.18.0; Apple M3 Max, 128 GiB.
* The 1 866-cell mesh: the cubic-voxel (shape-interpolation) arm of the anisotropic-voxel study, reconstructed
  with delaunay-watershed-3d 5bc1132 and foambryo 2da5186; only counts were read, nothing of it is in this
  repository.
* The cost anchors are medians quoted from the research records of both libraries (benchmark, 90-, 110-, 165- and
  1 866-cell tissues); most timings were taken on a shared machine and are provisional, peak memory is reliable.
* napari facts: napari's API reference and plugin guide, and PyPI, read on 11 October 2026 (napari 0.9.2 of
  29 September 2026, Python 3.11–3.14). No napari was installed or run for this phase.

## Reproduction (zsh, from the workspace root)

```zsh
export PYTHONPATH="$PWD/scratch/worktrees/the-foambryo-command/foambryo/src:$PWD/napari-foambryo/research/R124-napari-foambryo-plugin/scripts"
"$PWD/delaunay-watershed-3d/.venv/bin/python" "napari-foambryo/research/R124-napari-foambryo-plugin/scripts/compare_in_memory_and_rec_round_trip.py"
"$PWD/delaunay-watershed-3d/.venv/bin/python" "napari-foambryo/research/R124-napari-foambryo-plugin/scripts/check_mesh_survives_a_child_process.py"
"$PWD/delaunay-watershed-3d/.venv/bin/python" "napari-foambryo/research/R124-napari-foambryo-plugin/scripts/count_display_copy_sizes.py" "benchmark_case_000_reference_mesh=$PWD/benchmarking-dataset/000_mesh.rec"
"$PWD/delaunay-watershed-3d/.venv/bin/python" "napari-foambryo/research/R124-napari-foambryo-plugin/scripts/fit_reconstruction_cost_against_voxels.py"
```

The 1 866-cell count takes that mesh's `.npz` as `NAME=PATH`; its path is not recorded here.

## Limitations

* The bitwise route comparison and the child-process check ran on a four-cell fixture only; the specification
  makes the large cropped case a build gate (G5).
* The estimate's band is in-sample over seven anchors from one machine, not a validated interval.
* Nothing about napari was executed: every napari behaviour the specification relies on and did not read in the
  documentation is marked `TO VERIFY`.

## Status

Phase A complete; stopped for review. Open questions: `docs/SPECIFICATION.md` § 11.
