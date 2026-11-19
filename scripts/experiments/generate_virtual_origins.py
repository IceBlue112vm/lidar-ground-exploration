"""Generate virtual Patchwork++ sensor origins from RANSAC-inlier LAS.

Experiment 03, stage 1: origin placement only (no Patchwork++ execution).

1. Rebuild XY occupancy grid and connected components using the same rules as
   segment_ransac_patches.py.
2. Keep sufficiently large observed ground components.
3. Require every candidate to have a minimum grid boundary clearance.
4. Select one well-inside candidate per eligible component, then spread any
   remaining origins using farthest-point sampling, prioritizing large patches.
5. Anchor each origin to an ACTUAL inlier point within its selected grid cell.

The ground reference XYZ in origins.csv / ground_origins.xyz is the value to
pass to patchworkpp_ground.py --origin-ground X Y Z; that script adds the
sensor_height internally. Do NOT pass the sensor XYZ to --origin-ground.

Example:
  uv run python scripts/experiments/generate_virtual_origins.py \
    outputs/ransac_plane_001.las outputs/virtual_origins_05 \
    --cell-size 0.5 --min-points-per-cell 3 --min-area 10 \
    --clearance 1.0 --min-spacing 10 --max-origins 8
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from scipy import ndimage


FIELDS = [
    "origin_id", "component_rank", "component_id", "component_area_m2_approx",
    "component_inlier_points", "ground_x", "ground_y", "ground_z",
    "sensor_x", "sensor_y", "sensor_z", "clearance_m_approx",
]


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
    if not n:
        raise ValueError("The RANSAC-inlier LAS contains no points")

    x0, y0 = float(x.min()), float(y.min())
    nx = int(np.floor((float(x.max()) - x0) / args.cell_size)) + 1
    ny = int(np.floor((float(y.max()) - y0) / args.cell_size)) + 1
    if nx * ny > args.max_grid_cells:
        raise ValueError(
            f"XY grid {nx} x {ny} exceeds --max-grid-cells={args.max_grid_cells:,}"
        )

    ix = np.floor((x - x0) / args.cell_size).astype(np.int32)
    iy = np.floor((y - y0) / args.cell_size).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    flat = ix.astype(np.int64) * ny + iy

    counts = np.bincount(flat, minlength=nx * ny).reshape(nx, ny)
    occupied = counts >= args.min_points_per_cell
    structure = ndimage.generate_binary_structure(2, 1 if args.connectivity == 4 else 2)
    labels, component_count = ndimage.label(occupied, structure=structure)
    cell_counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    point_labels = labels.ravel()[flat]
    point_counts = np.bincount(point_labels, minlength=component_count + 1)

    components = [
        int(comp) for comp in range(1, component_count + 1)
        if cell_counts[comp] * args.cell_size**2 >= args.min_area
    ]
    components.sort(key=lambda comp: (-cell_counts[comp], -point_counts[comp]))
    components = components[:args.max_components]

    # A selected point is safely inside the observed patch only to the extent
    # justified by the chosen grid resolution/occupancy rule. This is a proxy,
    # NOT a guarantee of physical connectivity or traversability.
    candidate_sets = []
    for rank, comp in enumerate(components, 1):
        mask = labels == comp
        # Padding marks outside the observed grid as non-ground too.
        distance_cells = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        safe = mask & (distance_cells * args.cell_size >= args.clearance)
        ci, cj = np.nonzero(safe)
        if not len(ci):
            print(f"Skipping component #{rank}: no cells satisfy clearance {args.clearance:g}m")
            continue
        candidate_sets.append({
            "rank": rank,
            "id": comp,
            "area": float(cell_counts[comp] * args.cell_size**2),
            "points": int(point_counts[comp]),
            "i": ci,
            "j": cj,
            "cx": x0 + (ci + 0.5) * args.cell_size,
            "cy": y0 + (cj + 0.5) * args.cell_size,
            "clearance": distance_cells[ci, cj] * args.cell_size,
            "picked": set(),
            "seed_count": 0,
        })

    records = []
    selected_xy = []

    def pick_one(comp):
        # Use cell centers for ranking. Validate minimum spacing against the
        # actual inlier coordinates before accepting a candidate.
        candidate_count = len(comp["i"])
        if not candidate_count:
            return False
        if selected_xy:
            px = np.array([p[0] for p in selected_xy])
            py = np.array([p[1] for p in selected_xy])
            distances = np.hypot(
                comp["cx"][:, None] - px[None, :],
                comp["cy"][:, None] - py[None, :],
            ).min(axis=1)
        else:
            distances = np.full(candidate_count, np.inf)

        valid = distances >= args.min_spacing
        if comp["picked"]:
            valid[list(comp["picked"])] = False
        if not np.any(valid):
            return False

        # First seed in each component: largest boundary clearance.
        # Additional seeds: farthest from all previously selected origins.
        if comp["seed_count"] == 0:
            scores = comp["clearance"].copy()
        else:
            scores = distances + 0.01 * comp["clearance"]
        scores[~valid] = -np.inf

        while True:
            k = int(np.argmax(scores))
            if not np.isfinite(scores[k]):
                return False
            cell_i, cell_j = int(comp["i"][k]), int(comp["j"][k])
            cell_center_x, cell_center_y = float(comp["cx"][k]), float(comp["cy"][k])
            cell_index = cell_i * ny + cell_j
            matching = np.flatnonzero(flat == cell_index)
            if len(matching) == 0:
                raise RuntimeError("An occupied grid cell unexpectedly has no points")
            # Ensure chosen ground reference lies on an actually observed RANSAC inlier.
            dx = x[matching] - cell_center_x
            dy = y[matching] - cell_center_y
            point_index = int(matching[np.argmin(dx * dx + dy * dy)])
            gx, gy, gz = float(x[point_index]), float(y[point_index]), float(z[point_index])
            if selected_xy and min(
                np.hypot(gx - oldx, gy - oldy) for oldx, oldy in selected_xy
            ) < args.min_spacing:
                scores[k] = -np.inf
                continue

            comp["picked"].add(k)
            comp["seed_count"] += 1
            selected_xy.append((gx, gy))
            records.append({
                "origin_id": len(records) + 1,
                "component_rank": comp["rank"],
                "component_id": comp["id"],
                "component_area_m2_approx": round(comp["area"], 2),
                "component_inlier_points": comp["points"],
                "ground_x": round(gx, 4),
                "ground_y": round(gy, 4),
                "ground_z": round(gz, 4),
                "sensor_x": round(gx, 4),
                "sensor_y": round(gy, 4),
                "sensor_z": round(gz + args.sensor_height, 4),
                "clearance_m_approx": round(float(comp["clearance"][k]), 3),
            })
            return True

    # Coverage across disconnected observed patches: one seed each if feasible.
    for comp in candidate_sets:
        if len(records) >= args.max_origins:
            break
        pick_one(comp)

    # Spend remaining budget on larger patches first. This is a deliberately
    # simple baseline, not optimized for Patchwork++ segmentation quality.
    while len(records) < args.max_origins:
        added = False
        for comp in candidate_sets:
            if pick_one(comp):
                added = True
                break
        if not added:
            break

    if not records:
        raise RuntimeError(
            "No origins were eligible. Try a smaller --clearance, "
            "a smaller --min-area, or inspect the input ground patches."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "origins.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    for filename, fields in (
        ("ground_origins.xyz", ("ground_x", "ground_y", "ground_z")),
        ("sensor_origins.xyz", ("sensor_x", "sensor_y", "sensor_z")),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(" ".join(f"{rec[field]:.4f}" for field in fields) + "\n")

    print(f"Input RANSAC inliers: {n:,}")
    print(f"XY grid: {nx} x {ny}, cell size: {args.cell_size:g}m")
    print(f"Connected components: {component_count}")
    print(f"Qualified components (before clearance filtering): {len(components)}")
    print(f"Generated origins: {len(records)} / {args.max_origins}")
    for rec in records:
        print(
            f'  #{rec["origin_id"]:02d} [patch {rec["component_rank"]}] '
            f'ground=({rec["ground_x"]:.4f}, {rec["ground_y"]:.4f}, {rec["ground_z"]:.4f}), '
            f'clearance~{rec["clearance_m_approx"]:.2f}m'
        )
    print(f"Output: {args.output_dir}")
    print(f"Elapsed: {perf_counter() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="RANSAC inliers LAS (not the full source LAS)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-area", type=float, default=10.0)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--max-origins", type=int, default=8)
    parser.add_argument("--clearance", type=float, default=1.0, help="Minimum distance from patch boundary, m")
    parser.add_argument("--min-spacing", type=float, default=10.0, help="Minimum XY distance between origins, m")
    parser.add_argument("--sensor-height", type=float, default=1.723)
    parser.add_argument("--max-grid-cells", type=int, default=8_000_000)
    args = parser.parse_args()

    if (args.cell_size <= 0 or args.min_points_per_cell < 1
            or args.min_area <= 0 or args.max_components < 1
            or args.max_origins < 1 or args.clearance < 0
            or args.min_spacing <= 0 or args.sensor_height <= 0
            or args.max_grid_cells < 1):
        parser.error("Invalid nonpositive parameter: check sizes, counts, spacing, and height")
    run(args)


if __name__ == "__main__":
    main()
