# R135 — the viewers, second round (pointer)

The record of this item is in foambryo, on `research/R135-viewer-feedback-second-round`
(`research/R135-viewer-feedback-second-round/`: README, REPORT, HANDOFF, decisions, scripts, results, figures).

What it changed here, on `feature/first-version` (`50a4c8f`, on `c59151c`):

* the **tension colours** are foambryo's new map, `coolwarm` centred on the mean tension (blue - light grey - red),
  read from `foambryo.colour_scales`;
* an **undetermined tension**'s interface is not filled; a layer `<tensions> (undetermined: outlines)` draws it as a
  wireframe in the library's undetermined green; a selected row keeps it unfilled;
* **crop faces** are the library's glass (its fill at its opacity in the vertex colours, its edges as a wireframe),
  and are left out of the pressure layers;
* the **reconstruction estimate** is `foambryo.reconstruction_cost`'s: the plugin's copy of the fit is deleted;
* **`min_distance`** is on the Reconstruct form; the **docks** are listed Reconstruct, Infer, View, Export, Results;
* the layers' **display copies and colours are prepared in a worker thread** (after Reconstruct, Infer, a seed
  ensemble, and when explode leaves or returns to 0) and handed to napari on the GUI thread.

Plugin suite: 111 passed, 0 skipped (106 before; tiers T1-T3, gates G1-G11 unchanged). The specification's amendments are its § 13. The morning kit is refreshed:
`scratch/napari-foambryo/try_napari_foambryo.zsh` and `TRY-IT.md`.
