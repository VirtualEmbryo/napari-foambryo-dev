"""Does a foambryo mesh built in a spawned child process give bitwise the in-process tensions and pressures?

The plugin's proposed runner reconstructs in a child process (so a run can be cancelled by terminating it) and
sends the mesh back by pickling. Fixture: four Voronoi cells in a ball cut by the image box, z step 2x.
"""
import json
import multiprocessing as mp
import pickle
import time

import numpy as np

from compare_in_memory_and_rec_round_trip import bitwise_equal, four_cells_in_a_ball

SPACING = (2.0, 1.0, 1.0)


def reconstruct(mask):
    from foambryo import dcel_mesh_from_segmentation_mask
    return dcel_mesh_from_segmentation_mask(mask, spacing=SPACING)


if __name__ == "__main__":
    from foambryo import infer_forces
    mask = four_cells_in_a_ball(centre_z=6.0)
    here = reconstruct(mask)
    clock = time.perf_counter()
    with mp.get_context("spawn").Pool(1) as pool:
        there = pool.apply(reconstruct, (mask,))
    child_seconds = time.perf_counter() - clock
    t1, p1 = infer_forces(here)
    t2, p2 = infer_forces(there)
    print(json.dumps({
        "pickled_mesh_bytes": len(pickle.dumps(there)),
        "child_round_trip_seconds_including_spawn": round(child_seconds, 2),
        "same_points": bool(np.array_equal(here.v, there.v)),
        "same_triangles": bool(np.array_equal(here.f, there.f)),
        "tensions_bitwise_equal": bitwise_equal(t1, t2),
        "pressures_bitwise_equal": bitwise_equal(p1, p2),
    }, indent=1))
