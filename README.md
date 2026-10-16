# napari-foambryo

A napari plugin over delaunay-watershed-3d and foambryo: reconstruct the multimaterial mesh of a 3-D
segmentation, infer its relative surface tensions and cell pressures, show them on the tissue, and export the
same files as the `foambryo` command.

In development (0.1.0.dev0); nothing is released yet. Licence: GPL-3.0-or-later.

![Tensions of benchmark case 000 in napari](https://raw.githubusercontent.com/VirtualEmbryo/napari-foambryo/v0.1.0/docs/images/case_000_tensions.png)

*Benchmark case 000 (6 cells): the tensions of its outer interfaces on the reconstructed mesh (blue below the mean
tension, grey at it, red above), the colour bars and the table, in which the true tensions of this simulated
aggregate are shown beside the inferred ones. foambryo's Polyscope viewer shows the same colours for the same mesh.*

## What it does

* **Reconstruct**: a Labels layer, whoever made it (a file, napari's tools, another segmentation plugin), becomes a
  multimaterial mesh by delaunay-watershed-3d's default reconstruction, the one `foambryo mesh` runs. The voxel
  size comes from the layer, from another layer (the image that was segmented) or is typed: it is never guessed.
  Before the run the dock shows the estimated time and memory, and a report of the mask (cells, cells touching
  the image box, cells in several pieces, very small labels). The reconstruction runs in a separate process:
  **Cancel** stops it at once and frees its memory.
* **Infer**: the tensions, then the pressures, by foambryo's `infer_forces` with the options and defaults of
  `foambryo infer`; or a **seed ensemble** (`infer_tensions_over_seeds`), whose spread over the seeds is a lower
  bound on the error.
* **Results**: the tensions and pressures on the mesh, coloured as foambryo's own viewer colours them, the colour
  bars, and the tables. Select a row to see its interface or cell.
* **Export**: `mesh.rec`, `tensions.csv`, `pressures.csv`, `colour_bars.png` and `run.json`, the files of the
  `foambryo` command, with the command lines that reproduce them.

**One code path.** The plugin computes nothing of its own: every number is one that a public function of
foambryo returns for the same mask and options, the functions the `foambryo` command calls. The tests check this
bit for bit, and check that the exported files are byte for byte those of the command.

## Installation

napari-foambryo is not published yet. Once it is:

```bash
pip install "napari[all]" napari-foambryo
```

Until then, install it from a clone of its repository, in a virtual environment: `pip install "napari[all]"` and
`pip install ./napari-foambryo`. It needs delaunay-watershed-3d 0.7 and foambryo 0.7 (installed with it), and Python
3.11 or later.

## Quickstart

1. *File › Open Sample › foambryo › Benchmark case 000*.
2. *Plugins › foambryo › Reconstruct*: the layer and its voxel size are chosen; click **Reconstruct**.
3. *Plugins › foambryo › Infer*: click **Infer**. The tensions appear on the mesh.
4. *Plugins › foambryo › Results*: the colour bars and the tables; case 000's true tensions are in the table.
5. *Plugins › foambryo › Export*: choose a folder, click **Export**. If the folder already holds foambryo's files, a
   dialog lists them and asks before replacing them; other files are never touched.
6. *Plugins › foambryo › View*: **Explode** moves the cells apart (each interface drawn on both cells' sides), **Cut**
   clips everything with a plane normal to z, y or x, to see the interfaces inside the tissue (3-D view), and
   **Background** paints the canvas black, dark grey, light grey, white or a custom colour (a colour that one
   background hides, another shows).

The same, scripted: `python examples/quickstart.py` (add `--close-when-done` to close napari once the numbers are
printed).

## What the numbers mean

* **Tensions** are relative: they are gauged to a mean of 1 (*Advanced › mean tension*). Interfaces with no
  triple line are undetermined and reported as `nan`: not filled, drawn as green outlines by their own layer.
  The tension colours run from blue (low) through light grey (the mean tension) to red (high).
* **Pressures** are relative to the medium (label 0, *base pressure* 0), in units of tension per length of the
  voxel size's unit. Untick *pressures* in Infer for the tensions only (`foambryo infer --no-pressures`).
* **Diagnostics** (*Infer › Advanced › Diagnostics*: `basic`, or `spectrum` for the singular values too): the
  Results dock's Diagnostics tab lists every linear solve (rank, residual, condition number, …); Export writes them
  as `solves.csv` and `spectra.npz`.
* **Cut cells**: where cells reach the image box, the mesh closes them with *crop faces* (see-through glass with their edges); the
  pressures of the cells they cut are `nan`. Only pressure methods with a formulation for crop faces are offered.
* **Seed ensembles**: the seed standard deviation is a lower bound on the error, not an error bar.

## Large tissues

The estimate before a run is fitted to measured reconstructions (from a few seconds for a 6-cell aggregate to
about 13 minutes and 39 GiB for a 1 866-cell tissue at cubic voxels); its range is that of the fit, not a
guarantee. When the estimate exceeds the machine's memory, Reconstruct needs *run anyway* to be ticked: run
`foambryo mesh` on a larger machine instead (same numbers) and open the `.rec` file here.

## From a segmentation plugin to foambryo

Segmentation plugins (Cellpose, PlantSeg, micro-sam, napari-segment-blobs-and-things-with-membranes, …) leave a
Labels layer that often has no voxel size, and may label the background with something else than 0. In the
Reconstruct dock:

* **Voxel size from › layer: …** the image that was segmented (or *typed*);
* **Medium is label …** the background's label: a copy of the layer with that label and 0 swapped is made and
  reconstructed (and exported, since the equivalent command needs it);
* read the **mask report**: a background split into several pieces shows up as extra cells touching the image
  box. Merge them into the background with napari's fill tool, then Reconstruct.

The results are marked *out of date* if the labels change after the reconstruction, and Export waits for a new
one. Example: `python examples/from_a_segmentation_plugin.py` (with `--close-when-done`, it clicks Reconstruct and
Infer itself and closes napari).

![A membrane segmentation plugin's labels, with the mask report](https://raw.githubusercontent.com/VirtualEmbryo/napari-foambryo/v0.1.0/docs/images/from_a_segmentation_plugin.png)

## Reproducing a result without napari

`run.json` lists the command lines that rebuild the exported folder, for example

```bash
foambryo mesh mask.npz -o mesh.rec --spacing 1.0 1.0 1.0 --spacing-unit voxel --min-distance 3
```

```bash
foambryo infer mesh.rec -o . --tension-method young-dupre --pressure-method variational
```

(Tick *include the mask* in Export to get `mask.npz`.)

## Credits, contact, citations

If you use this tool, please cite the associated article: Ichbiah S., Delbary F., McDougall A., Dumollard R.,
Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature
Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7).
Benchmark case 000, shipped as sample data with its true tensions, comes from that article's benchmark.

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon
(Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues
Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding
defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on
large tissues.

## Licence

GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`); the text is in
`LICENSE`. How the plugin works, dock by dock:
[docs/how-it-works.md](https://github.com/VirtualEmbryo/napari-foambryo/blob/v0.1.0/docs/how-it-works.md).
