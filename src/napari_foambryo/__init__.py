"""napari-foambryo: surface tensions and cell pressures of a 3-D segmentation, in napari.

A napari plugin over delaunay-watershed-3d (the multimaterial mesh of a segmentation) and foambryo (relative surface
tensions and cell pressures from the mesh's geometry). Every number it shows or writes is one a public function of
the two libraries returns for the same inputs, the same functions the `foambryo` command calls; it computes nothing
scientific of its own. How it works, dock by dock: `docs/how-it-works.md`.
"""

__version__ = "0.1.0.dev0"
