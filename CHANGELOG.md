# Changelog

User-facing changes only.

## Unreleased (planned for 0.1.1)

Nothing yet.

## 0.1.0 — 2026-10-17

The first release of napari-foambryo, a napari plugin over delaunay-watershed-3d 0.7.0 and foambryo 0.7.0, released with them.

- **Five docks**, under *Plugins › foambryo*: **Reconstruct** (a Labels layer and its voxel size become a multimaterial mesh, by the reconstruction `foambryo mesh` runs, with a report of the mask and an estimate of the time and memory before the run), **Infer** (tensions, then pressures, as `foambryo infer`; or a seed ensemble, as `foambryo infer-over-seeds`), **Results** (the tensions and pressures on the mesh, in foambryo's colours, with their colour bars and tables; selecting a row shows its interface or cell), **View** (explode, cut, background) and **Export** (the files of the `foambryo` command, with the command lines that rebuild them).
- **One code path**: every number shown or written is one a public function of foambryo returns for the same mask and options; the tests check this bit for bit, and that the exported files are byte for byte the command's.
- **Runs in a separate process**: napari stays responsive, and Cancel stops a reconstruction, an inference or a seed ensemble at once and frees its memory. When the estimate exceeds the machine's memory, Reconstruct asks for *run anyway*.
- **Masks from other plugins**: the voxel size can be taken from another layer, the medium's label can be another value than 0, and results are marked out of date when the labels change.
- **Cropped tissues and anisotropic voxels**, as in foambryo 0.7.0: crop faces drawn as glass, cut cells' pressures `nan`; voxels longer along one axis resampled to cubic voxels before meshing.
- A reader and a writer for `.rec` meshes, and sample data: benchmark case 000 (6 cells, with its true tensions) and three small generated foams.
- Requires Python 3.11 or later, napari 0.9 or later, delaunay-watershed-3d 0.7.0 and foambryo 0.7.0 or later.
- **Dependencies**: the minimum versions are the oldest with which the test suite passes: napari 0.9 (0.8 lacks a call the plugin makes), numpy 2.0.0 and matplotlib 3.8.4.

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.
