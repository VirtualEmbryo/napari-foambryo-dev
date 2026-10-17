# How napari-foambryo works

napari-foambryo adds five docks to napari, under *Plugins › foambryo*, plus a reader, a writer and sample data. The plugin computes nothing scientific of its own: each dock calls a public function of delaunay-watershed-3d (dw3d) or foambryo, the same function the `foambryo` command calls, with the same defaults. A result of the plugin is therefore the result of the command for the same mask and options.

## The docks

| dock | what it does | the library function it calls |
|---|---|---|
| **Reconstruct** | Takes a Labels layer and its voxel size, checks the mask, reports on it, estimates the time and memory, then builds the multimaterial mesh. | `foambryo.segmentation_masks.check_segmentation_mask` and `segmentation_mask_report` (the checks and the report); `foambryo.reconstruction_cost.estimate_reconstruction_cost` (the estimate); `foambryo.io.reconstruct_mask`, dw3d's default reconstruction, the one `foambryo mesh` runs (resampled to cubic voxels first when the voxels are longer along one axis) |
| **Infer** | Infers the tensions, then the pressures, of the reconstructed mesh or of a `.rec` file opened in napari; or a seed ensemble of the mask. | `foambryo.infer_tensions_then_pressures` (the two calls of `infer_forces`, as `foambryo infer`; `infer_tensions` alone when *pressures* is unticked); `foambryo.infer_tensions_over_seeds` for a seed ensemble, as `foambryo infer-over-seeds` |
| **Results** | Shows the tables of interfaces, cells and linear solves, the colour bars and the warnings; selecting a row shows its interface or cell. | the values Infer returned; the colours and ranges of `foambryo.colour_scales`, those of foambryo's Polyscope viewer |
| **View** | Explodes the cells apart, cuts the tissue with a plane normal to z, y or x, and sets the canvas's background. Nothing is computed again. | `DcelData.compute_scattered_arrays` (the displacement of the Polyscope viewer's scattered view); napari's clipping planes; `foambryo.colour_scales.BACKGROUND_COLOURS` |
| **Export** | Writes the files of the `foambryo` command into a folder, with the command lines that rebuild them. | the writers of `foambryo.results`; `foambryo.colour_scales.save_colour_bars` |

The reader opens a `.rec` or `.arec` mesh file (`foambryo.dcel_mesh_from_file`); the writer saves a mesh layer as `.rec` (*File › Save Layer*). The sample data are benchmark case 000 (6 cells, with its true tensions) and three small generated foams (cubic voxels, voxels twice as long along z, and a foam cut by the image box). In the layer list, a Labels layer's right-click menu offers *Reconstruct with foambryo*, which opens Reconstruct on that layer.

## Where the work runs

A reconstruction, an inference and a seed ensemble each run in a separate process, so that napari stays responsive and **Cancel** stops the run at once and frees its memory. A cancelled or failed run adds nothing. The estimate shown before a run is foambryo's own, a fit to measured reconstructions, not a guarantee: when it exceeds the machine's memory, Reconstruct asks you to tick *run anyway*, or to run `foambryo mesh` on a larger machine and open the `.rec` file here.

## Where the results go

* **Into napari**, as layers: the mesh over the Labels layer, the tensions (one colour per interface), the pressures (one colour per cell), and, when they exist, the undetermined tensions (green outlines) and the crop faces (see-through glass). Each layer is placed exactly on the voxels of its Labels layer.
* **Into the Results dock**: the tables, which *Export CSV* writes one at a time.
* **Into a folder, only when you click Export**: `mesh.rec`, `tensions.csv`, `pressures.csv`, `colour_bars.png` and `run.json` (with diagnostics, `solves.csv` and `spectra.npz`; for a seed ensemble, a `seed-ensemble/` folder; on request, `mask.npz`). They are the files the `foambryo` command writes for the same mask and options, and `run.json` lists the command lines that rebuild them. A folder that already holds such files is not changed without your confirmation, and no other file in it is ever touched.

Nothing else is written: the plugin keeps its runs in memory until napari is closed.

## What it leaves out

The plugin offers what the `foambryo` command offers, no more: not the reconstruction seed (use a seed ensemble), the faces of the image box where the tissue really ends, the resampling method, the angle reading of a single mesh, the wedge refusal or the error bars of `foambryo.infer_tensions_with_uncertainty`. Use foambryo from Python for those.
