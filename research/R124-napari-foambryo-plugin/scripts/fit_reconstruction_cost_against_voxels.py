"""Fit reconstruction wall time and peak memory against the voxels handed to dw3d, from measured anchor points.

Anchor points: per-reconstruction medians quoted in foambryo and dw3d research records (default reconstruction,
min_distance 3, relocation on, Apple M3 Max 128 GiB). Timings mostly on a shared machine (provisional); peak RSS
reliable. Prints the power-law fits and each point's ratio measured / predicted.
"""
import json

import numpy as np

# name, voxels handed to dw3d, seconds, peak GiB, source
POINTS = [
    ("benchmark at real cell size (median)", 1.5e6, 0.8, 0.6, "R121 SUMMARY_TABLES"),
    ("benchmark default size (median)", 7.72e6, 2.3, 1.2, "R116 REPORT"),
    ("90-cell simulated tissue", 4.2e7, 21.0, 4.7, "R111 csv / R116"),
    ("110-cell tissue, native grid", 7.99e7, 35.0, 5.1, "R116 REPORT"),
    ("165-cell simulated tissue", 1.17e8, 76.0, 11.6, "R111 csv / R116"),
    ("1 866-cell tissue, native grid", 4.98e8, 610.0, 29.1, "R116 REPORT"),
    ("1 866-cell tissue, cubic voxels", 8.3e8, 807.0, 45.6, "R121 SUMMARY_TABLES"),
]
v = np.log([p[1] for p in POINTS])
out = {}
for column, name in ((2, "seconds"), (3, "peak_GiB")):
    y = np.log([p[column] for p in POINTS])
    slope, intercept = np.polyfit(v, y, 1)
    ratio = np.exp(y - (slope * v + intercept))
    out[name] = {"exponent": round(slope, 3), "prefactor": float(np.exp(intercept)),
                 "ratio_measured_over_predicted": {
                     p[0]: round(float(r), 2) for p, r in zip(POINTS, ratio, strict=True)},
                 "max_factor_error": round(float(np.max(np.maximum(ratio, 1 / ratio))), 2)}
print(json.dumps(out, indent=1))
