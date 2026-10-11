"""Count the vertices and triangles of the display copies a napari Surface layer would need for a multimaterial mesh.

napari's Surface layer colours vertices, not faces. To give every interface (tensions) or every cell (pressures) its
own uniform colour, the display mesh duplicates each vertex once per interface (or per cell side) it belongs to.
This script counts those copies on a saved mesh and prints the array sizes, as JSON, without writing anything.

Usage: python count_display_copy_sizes.py NAME=MESH.npz|MESH.rec [NAME=...]
A .npz holds `points`, `triangles`, `labels` and optionally `crop_faces`; a .rec is read with dw3d.io.
Only counts are printed: no coordinate, label or path.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.endswith(".rec"):
        from dw3d.io import load_rec_with_crop_faces

        points, triangles, labels, crop = load_rec_with_crop_faces(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            points, triangles, labels = data["points"], data["triangles"], data["labels"]
            crop = data["crop_faces"] if "crop_faces" in data.files else None
    if crop is None:
        crop = np.zeros(len(triangles), dtype=bool)
    return np.asarray(points), np.asarray(triangles, dtype=np.int64), np.asarray(labels, dtype=np.int64), crop


def unique_pairs(group: np.ndarray, vertex: np.ndarray) -> int:
    """Number of distinct (group, vertex) pairs."""
    keys = group.astype(np.int64) * (int(vertex.max()) + 1) + vertex.astype(np.int64)
    return int(np.unique(keys).size)


def counts(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray, crop: np.ndarray) -> dict:
    n_vertices, n_triangles = len(points), len(triangles)
    pair = np.sort(labels, axis=1)
    interface_keys, interface_id = np.unique(pair[:, 0] * (int(pair.max()) + 1) + pair[:, 1], return_inverse=True)
    cells = np.unique(labels[labels > 0])

    # Tensions: one copy of a vertex per interface that uses it (crop faces form their own grey group).
    group = np.where(crop, -1, interface_id) + 1
    tension_vertices = unique_pairs(np.repeat(group, 3), triangles.ravel())

    # Pressures: each triangle once per side that is a cell (label > 0), one vertex copy per cell that uses it.
    sides = [(labels[:, j] > 0) for j in (0, 1)]
    cell_of_side = np.concatenate([labels[sides[0], 0], labels[sides[1], 1]])
    tri_of_side = np.concatenate([triangles[sides[0]], triangles[sides[1]]])
    pressure_vertices = unique_pairs(np.repeat(cell_of_side, 3), tri_of_side.ravel())
    pressure_triangles = int(len(tri_of_side))

    # Pressures as `foambryo view` draws its surface: each triangle once, in its higher-labelled material's colour.
    surface_group = labels.max(axis=1)
    pressure_surface_vertices = unique_pairs(np.repeat(surface_group, 3), triangles.ravel())

    return {
        "vertices": n_vertices,
        "triangles": n_triangles,
        "interfaces": int(interface_keys.size),
        "cells": int(cells.size),
        "crop_faces": int(np.count_nonzero(crop)),
        "tension_copy_vertices": tension_vertices,
        "tension_copy_triangles": n_triangles,
        "pressure_copy_vertices": pressure_vertices,
        "pressure_copy_triangles": pressure_triangles,
        "pressure_surface_copy_vertices": pressure_surface_vertices,
        "pressure_surface_copy_triangles": n_triangles,
        "unindexed_vertices": 3 * n_triangles,
    }


def megabytes(c: dict) -> dict:
    """Bytes of the arrays a Surface layer is handed: float64 vertices and values, int64 faces, float32 RGBA colours."""
    def layer(v: int, f: int) -> float:
        return (v * (3 * 8 + 8 + 4 * 4) + f * 3 * 8) / 2**20

    return {
        "plain_mesh_MiB": round(layer(c["vertices"], c["triangles"]), 1),
        "tension_copy_MiB": round(layer(c["tension_copy_vertices"], c["tension_copy_triangles"]), 1),
        "pressure_copy_MiB": round(layer(c["pressure_copy_vertices"], c["pressure_copy_triangles"]), 1),
        "pressure_surface_copy_MiB": round(
            layer(c["pressure_surface_copy_vertices"], c["pressure_surface_copy_triangles"]), 1),
        "unindexed_copy_MiB": round(layer(c["unindexed_vertices"], c["triangles"]), 1),
    }


if __name__ == "__main__":
    out = {}
    for argument in sys.argv[1:]:
        name, path = argument.split("=", 1)
        c = counts(*load(path))
        c["tension_copy_vertex_ratio"] = round(c["tension_copy_vertices"] / c["vertices"], 3)
        c["pressure_copy_vertex_ratio"] = round(c["pressure_copy_vertices"] / c["vertices"], 3)
        out[name] = {**c, **megabytes(c)}
    print(json.dumps(out, indent=1))
