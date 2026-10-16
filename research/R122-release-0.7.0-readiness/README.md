# R122 — 0.7.0 readiness (pointer)

The record of this cross-project item is on foambryo, branch `research/R122-release-0.7.0-readiness`, folder
`research/R122-release-0.7.0-readiness/`.

What concerns this repository (`feature/first-version` cdefce1, unchanged):

* The wheel builds and passes `twine check`; its manifest validates (`npe2 validate`); its headless suite passed 118 of
  118 against the 0.7.0 wheels of dw3d and foambryo. It cannot be installed with its dependencies until both
  libraries are versioned 0.7.0 (they still read 0.6.0).
* Before the repository is public: `docs/SPECIFICATION.md` is an internal design record (approvals, decisions,
  evidence labels, hashes) linked from the README and `__init__`; three source comments cite its decisions ("decision
  Qn"); the `opengl` pytest marker's help says "tier T4". No Project-URL in the metadata; the README's images are
  relative links, which PyPI does not show.
* The ensemble estimate (per-seed time × rounds) assumes workers do not slow each other: measured in the remaining
  quiet run, not yet. The refit of foambryo's estimate (not recommended) would make two of this plugin's tests fail.
