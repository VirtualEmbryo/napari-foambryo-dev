"""Quickstart: benchmark case 000 from its mask to tensions and pressures, in napari.

Opens napari with the plugin's docks and the sample *benchmark case 000*, then clicks Reconstruct and Infer for you
(the same buttons you would click), and prints the median error against the case's true tensions. Needs a display.

    python examples/quickstart.py
    python examples/quickstart.py --close-when-done     # closes napari once the numbers are printed
"""

import sys

import napari
from qtpy.QtCore import QTimer

from napari_foambryo._widgets import ResultsWidget


def main() -> None:
    viewer = napari.Viewer(title="napari-foambryo quickstart")
    _, reconstruct = viewer.window.add_plugin_dock_widget("napari-foambryo", "Reconstruct")
    _, infer = viewer.window.add_plugin_dock_widget("napari-foambryo", "Infer")
    results = ResultsWidget(viewer)
    viewer.window.add_dock_widget(results, name="Results", area="right")
    viewer.open_sample("napari-foambryo", "benchmark_case_000")
    viewer.dims.ndisplay = 3
    reconstruct.select_layer("benchmark case 000")

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
        print(results.summary.text())
        if "--close-when-done" in sys.argv:
            viewer.close()

    reconstruct.start()  # the Reconstruct button
    QTimer.singleShot(500, after_reconstruct)
    napari.run()


if __name__ == "__main__":
    main()
