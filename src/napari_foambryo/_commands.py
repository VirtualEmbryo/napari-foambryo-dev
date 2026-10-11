"""The Layers menu's *Reconstruct with foambryo*: the selected Labels layer, in the Reconstruct widget.

napari 0.9 does not let plugins add to the layer list's right-click menu (`napari/layers/context` is not a
contributable menu), so the entry is under Layers › Measure and acts on the selected layer.
"""

import napari
from napari.layers import Labels

PLUGIN = "napari-foambryo"


def reconstruct_selected_layer(viewer: napari.viewer.Viewer) -> None:
    """Show the Reconstruct widget with the selected Labels layer chosen."""
    _, widget = viewer.window.add_plugin_dock_widget(PLUGIN, "Reconstruct")
    layer = viewer.layers.selection.active
    if isinstance(layer, Labels):
        widget.select_layer(layer.name)
