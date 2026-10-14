"""From a segmentation plugin to foambryo: a Labels layer as another plugin leaves it.

Segmentation plugins often leave a Labels layer that (1) has no voxel size, although the image it was made from
has one, and (2) labels the background with something else than 0. This example builds such a pair of layers from
the generated foam (an image layer with a voxel size, and labels with the background at 1 and no scale), then sets
the Reconstruct dock as you would: *Voxel size from › layer: membranes*, *Medium is label 1*. The mask report
appears under the form before you click Reconstruct. Needs a display.

    python examples/from_a_segmentation_plugin.py
    python examples/from_a_segmentation_plugin.py --close-when-done   # clicks Reconstruct and Infer, then closes
"""

import sys

import napari
import numpy as np
from qtpy.QtCore import QTimer

from napari_foambryo._sample_data import generated_foam


def main() -> None:
    foam = generated_foam()[::2]  # voxels twice as long along z
    membranes = np.zeros(foam.shape, dtype=np.uint8)
    for axis in range(3):  # bright where the label changes: a stand-in for a membrane stain
        membranes |= (np.diff(foam, axis=axis, prepend=np.take(foam, [0], axis=axis)) != 0).astype(np.uint8) * 255
    labels = np.where(foam == 0, 1, foam + 1).astype(np.uint16)  # the background is label 1

    viewer = napari.Viewer(title="napari-foambryo: from a segmentation plugin")
    viewer.add_image(membranes, name="membranes", scale=(1.0, 0.5, 0.5), units="um")
    viewer.add_labels(labels, name="segmentation")  # no voxel size, as the plugin left it
    _, reconstruct = viewer.window.add_plugin_dock_widget("napari-foambryo", "Reconstruct")
    _, infer = viewer.window.add_plugin_dock_widget("napari-foambryo", "Infer")
    reconstruct.select_layer("segmentation")
    reconstruct.spacing_from.setCurrentText("layer: membranes")
    reconstruct.medium_box.setValue(1)
    viewer.dims.ndisplay = 3
    print("Read the voxel size and the mask report in the Reconstruct dock, then click Reconstruct, then Infer.")
    if "--close-when-done" in sys.argv:  # the same clicks, made for you, for a run without a person

        def after_reconstruct() -> None:
            if reconstruct._worker is not None:
                QTimer.singleShot(500, after_reconstruct)
                return
            print(reconstruct.status.text())
            infer.start()
            QTimer.singleShot(500, after_infer)

        def after_infer() -> None:
            if infer._worker is not None:
                QTimer.singleShot(500, after_infer)
                return
            print(infer.status.text())
            viewer.close()

        reconstruct.start()
        QTimer.singleShot(500, after_reconstruct)
    napari.run()


if __name__ == "__main__":
    main()
