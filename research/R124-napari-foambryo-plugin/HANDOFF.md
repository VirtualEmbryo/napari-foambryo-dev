# R124 phase B — handoff (12 October 2026, morning)

## To try it

From the workspace root, in zsh:

```zsh
zsh "scratch/napari-foambryo/try_napari_foambryo.zsh"
```

The guide is `scratch/napari-foambryo/TRY-IT.md` (five walks, what is not finished, known problems).

## Repositories, branches, commits (nothing pushed, merged or tagged)

* **foambryo** `feature/public-helpers-for-front-ends`, on `feature/the-foambryo-command` (669a976), worktree
  `scratch/worktrees/public-helpers-for-front-ends/foambryo`: ac8cd0c (crop-face property), 6daeebe (progress
  callback), ce2125a (public helpers; the command calls them), c274e54 (`foambryo mesh` prints the mask report;
  the one changed assertion). Not merged into the command's branch: it waits for the command's review, then is
  rebased and merged (decision Q1). c274e54 can be dropped alone.
* **napari-foambryo** `feature/first-version`, on `main` (a90721f), worktree
  `scratch/worktrees/napari-foambryo-first-version/napari-foambryo`: adcb9f4 … 7613194. `main` is to be
  fast-forwarded by Hervé (tip 7613194).
* **napari-foambryo** `research/R124-napari-foambryo-plugin` (the shared checkout `napari-foambryo/`): the
  specification version 1, this record.
* **delaunay-watershed-3d**: unchanged; a detached worktree at `master` 68eccf4
  (`scratch/worktrees/napari-foambryo-dw3d-master/`) serves the scratch venv's editable install.
* **Scratch** (not committed, by design): `scratch/napari-foambryo/` holds the venv (napari 0.9.2, PyQt6 6.11,
  NumPy 2.5.4, SciPy 1.18.1, Python 3.12.0; the segmentation plugin of the manual check), the morning kit, the
  screenshots, the exports and the logs.

## Scientific impact

None on any number: the plugin calls the libraries' public functions, and the helpers branch changes no number
(the command's tests pass unchanged; the suites agree). What is new is a way to reach the method from napari, and
one measurement: on case 000, a membrane segmentation plugin's labels cost 0.03 of median relative tension error
against the true mask, almost all on one interface (REPORT, Figure 1).

## Tests run, and not run

Run: the foambryo suite at 669a976 and at the helpers tip; the plugin's suite (87, tiers T1–T3, gates G1–G11);
ruff on both; `ty` on foambryo; the scanners; the real-window runs (REPORT). Not run: tier T4 as a test (no
OpenGL under Qt's offscreen platform; the real-window runs stand in for it); delaunay-watershed-3d's suite (no
change there); Python 3.11 and 3.13; napari other than 0.9.2; a 1 866-cell tissue (the real-size
smoke run was a 90-cell tissue, REPORT).

## Reproduction (zsh, from the workspace root)

```zsh
cd "scratch/worktrees/napari-foambryo-first-version/napari-foambryo"
```

```zsh
QT_QPA_PLATFORM=offscreen "../../../napari-foambryo/venv/bin/pytest" -q -p no:cacheprovider tests
```

```zsh
cd "../../public-helpers-for-front-ends/foambryo"
```

```zsh
PYTHONPATH="${PWD}/src" MPLBACKEND=Agg "../../../../foambryo/.venv/bin/pytest" -q -p no:cacheprovider tests
```

The manual checks: `research/R124-napari-foambryo-plugin/scripts/manual_check_from_a_membrane_segmentation_plugin.py`
and `smoke_run_on_a_cropped_foam_and_a_90_cell_tissue.py` (their docstrings give the arguments; they open a napari
window and close it when done).

## Root check

`planning/scripts/check_workspace_root.py`: OK at the start, after each milestone, and at the end.

## Decisions to look at first

`phase_b_decision_log.md`: 3 (the mask-report commit, c274e54), 7 (no layer right-click menu in napari 0.9:
*Layers › Measure* instead), 9 (the generated foam's size), 13 (layers hidden in 3-D), 20 (the background in
pieces).

## `TO VERIFY`

See REPORT, last section.
