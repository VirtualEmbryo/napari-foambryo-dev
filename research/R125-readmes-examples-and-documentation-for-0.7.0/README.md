# R125 — READMEs, examples and documentation for 0.7.0 (pointer)

The record of this item is in foambryo, on `research/R125-readmes-examples-and-documentation-for-0.7.0`
(`research/R125-readmes-examples-and-documentation-for-0.7.0/`: README, REPORT, HANDOFF, scripts, results, figures,
screenshots).

What it changed here, on `feature/first-version` (`e3f34d4`, `a2827d7`, added on `5a84dfd`):

* the two examples take `--close-when-done`: the quickstart closes napari once it has printed its numbers, and
  *From a segmentation plugin* clicks Reconstruct and Infer itself, then closes (without the flag nothing changes);
  both ran so on the Mac's display (case 000: median relative error 0.041 against its true tensions);
* the README says the plugin is not published yet and how to install it from a clone; its screenshot of benchmark
  case 000 is taken again in the current colours (coolwarm tensions, the same bar as foambryo's Polyscope window);
  the walk-through GIF, made with the earlier colours, is no longer shown (the file stays);
* `CITATION.cff`, naming the article to cite.

Plugin suite at a2827d7: 114 passed, 0 skipped (= 5a84dfd).
