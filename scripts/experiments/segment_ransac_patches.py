"""Split RANSAC plane inliers into XY-connected surface patches.

Usage:
  uv run python scripts/experiments/segment_ransac_patches.py \
      outputs/ransac_plane_001.las outputs/ransac_patches_05 \
      --cell-size 0.5 --min-points-per-cell 3 --min-area 10 --export-top 5

The input is the RANSAC inlier LAS, not the original complete LAS.
The output seed XYZs are GROUND REFERENCE POINTS, not sensor XYZs.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from scipy import ndimage


def run(args):
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.output_dir}")

    t0 = perf_counter()
    las = laspy.read(args.input)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    n = len(x)
    if n == 0:
        raise ValueError("Input LAS is empty")

    x0, y0 = float(x.min()), float(y.min())
    nx = int(np.floor((float(x.max()) - x0) / args.cell_size)) + 1
    ny = int(np.floor((float(y.max()) - y0) / args.cell_size)) + 1
    ncells = nx * ny
    if ncells > args.max_grid_cells:
        raise ValueError(
            f"Grid would be {nx} x {ny} = {ncells:,} cells. "
            f"Increase --cell-size or --max-grid-cells (memory caution)."
        )

    ix = np.floor((x - x0) / args.cell_size).astype(np.int32)
    iy = np.floor((y - y0) / args.cell_size).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    flat = ix.astype(np.int64) * ny + iy

    counts = np.bincount(flat, minlength=ncells).reshape(nx, ny)
    occupied = counts >= args.min_points_per_cell
    if args.connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        structure = ndimage.generate_binary_structure(2, 2)
    labels, number = ndimage.label(occupied, structure=structure)

    cell_counts = np.bincount(labels.ravel(), minlength=number + 1)
    point_labels = labels.ravel()[flat]
    point_counts = np.bincount(point_labels, minlength=number + 1)
    candidates = [
        int(k) for k in range(1, number + 1)
        if cell_counts[k] * args.cell_size**2 >= args.min_area
    ]
    candidates.sort(key=lambda k: (-cell_counts[k], -point_counts[k]))

    # Prefer an interior cell far from the observed patch boundaries.
    # This is a grid-based heuristic, not proof of actual floor continuity.
    distance = ndimage.distance_transform_edt(np.pad(occupied, 1))[1:-1, 1:-1]
    records = []
    for rank, comp in enumerate(candidates, 1):
        center_i, center_j = ndimage.maximum_position(
            distance, labels=labels, index=comp
        )
        local = (ix == center_i) & (iy == center_j)
        if not local.any():
            raise RuntimeError("Selected grid cell has no source points")
        seed_x = x0 + (center_i + 0.5) * args.cell_size
        seed_y = y0 + (center_j + 0.5) * args.cell_size
        ground_z = float(np.median(z[local]))
        records.append({
            "rank": rank,
            "component_id": comp,
            "occupied_cells": int(cell_counts[comp]),
            "area_m2_approx": round(float(cell_counts[comp] * args.cell_size**2), 2),
            "inlier_points": int(point_counts[comp]),
            "ground_x": round(seed_x, 4),
            "ground_y": round(seed_y, 4),
            "ground_z": round(ground_z, 4),
            "sensor_x": round(seed_x, 4),
            "sensor_y": round(seed_y, 4),
            "sensor_z": round(ground_z + args.sensor_height, 4),
            "interior_clearance_m_approx": round(float(distance[center_i, center_j] * args.cell_size), 2),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "patches.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "rank", "component_id", "occupied_cells", "area_m2_approx", "inlier_points",
            "ground_x", "ground_y", "ground_z", "sensor_x", "sensor_y", "sensor_z",
            "interior_clearance_m_approx",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    with (args.output_dir / "ground_seeds.xyz").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(f'{row["ground_x"]:.4f} {row["ground_y"]:.4f} {row["ground_z"]:.4f}\n')

    # Optionally write the largest observed patches as separate LAS files.
    for row in records[: args.export_top]:
        part = laspy.LasData(las.header)
        part.points = las.points[point_labels == row["component_id"]].copy()
        output = args.output_dir / f'patch_{row["rank"]:02d}.las'
        part.write(output)
        print(f'Exported {output}: {len(part.points):,} points', flush=True)

    print(f"Input inliers: {n:,}")
    print(f"Grid: {nx} x {ny} cells ({args.cell_size}m)")
    print(f"Connected components: {number:,}")
    print(f"Patches >= {args.min_area:g} m2: {len(records):,}")
    print(f"Points in under-populated cells: {int(point_counts[0]):,}")
    for row in records[:10]:
        print(
            f'  #{row["rank"]}: {row["area_m2_approx"]:.1f} m2, '
            f'{row["inlier_points"]:,} points, '
            f'ground=({row["ground_x"]}, {row["ground_y"]}, {row["ground_z"]})'
        )
    print(f"Output: {args.output_dir}")
    print(f"Elapsed: {perf_counter() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-area", type=float, default=10.0)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--sensor-height", type=float, default=1.723)
    parser.add_argument("--export-top", type=int, default=5)
    parser.add_argument("--max-grid-cells", type=int, default=8_000_000)
    args = parser.parse_args()
    if args.cell_size <= 0 or args.min_points_per_cell < 1 or args.min_area <= 0:
        parser.error("cell-size, min-points-per-cell and min-area must be positive")
    if args.export_top < 0:
        parser.error("export-top must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()

