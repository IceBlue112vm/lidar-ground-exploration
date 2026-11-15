"""Experiment 04: one-step ground exploration beyond the initial RANSAC plane.

Two deliberate stages (NO ground-result union):

  discover: From origin_01_ground.las, find approximately horizontal, spatially
            connected *candidate* patches at heights outside the initial
            RANSAC plane. Export patch previews and candidate sensor origins.
  run:      Run Patchwork++ independently at ONE reviewed candidate origin and
            at the original origin #1, and export the newly classified points
            relative to origin #1 as a DIAGNOSTIC DIFFERENCE, not a union.

The 3D voxel / local-horizontal-support filter is a heuristic, not a semantic
floor detector. Horizontal roofs and platforms can pass; review preview LAS
files in CloudCompare before choosing an origin.

Example:
  uv run python src/pointcloud_lab/recursive_ground_step1.py discover \
    outputs/multi_origin_08/origin_01_ground.las \
    outputs/ransac_plane_001.las \
    outputs/virtual_origins_05/origins.csv \
    outputs/recursive_04_candidates

  uv run python -u src/pointcloud_lab/recursive_ground_step1.py run \
    data/SKV1.las \
    outputs/virtual_origins_05/origins.csv \
    outputs/recursive_04_candidates/origins.csv \
    outputs/recursive_04_test_01 --origin-id 1
"""

import argparse
import csv
import gc
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from scipy import ndimage


def require_new_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no origin rows: {path}")
    return rows


def read_sensor_xyz(row):
    xyz = np.array([float(row[k]) for k in ("sensor_x", "sensor_y", "sensor_z")], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite origin in row: {row}")
    return xyz


def fit_ransac_plane(inlier_las: Path, chunk_size: int):
    """Fit z = ax + by + c using samples of *existing* RANSAC inliers.

    No fresh RANSAC search is performed. The input is used only to distinguish
    patches at genuinely new elevations in this FIRST expansion experiment.
    """
    sample_xyz = []
    with laspy.open(inlier_las) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            stride = max(1, len(chunk) // 4000)
            sl = slice(None, None, stride)
            sample_xyz.append(np.column_stack((np.asarray(chunk.x)[sl],
                                               np.asarray(chunk.y)[sl],
                                               np.asarray(chunk.z)[sl])))
    if not sample_xyz:
        raise ValueError(f"Empty RANSAC inlier LAS: {inlier_las}")
    sample = np.concatenate(sample_xyz, axis=0)
    A = np.column_stack((sample[:, 0], sample[:, 1], np.ones(len(sample))))
    abc, _, _, _ = np.linalg.lstsq(A, sample[:, 2], rcond=None)
    slope_deg = float(np.degrees(np.arctan(np.hypot(abc[0], abc[1]))))
    if slope_deg > 25:
        raise ValueError(f"Initial RANSAC plane is too tilted for z-height comparison ({slope_deg:.1f} deg)")
    return abc, slope_deg


def voxel_shape(header, xy_cell: float, z_cell: float, max_voxels: int):
    mins = np.asarray(header.mins, dtype=np.float64)
    maxs = np.asarray(header.maxs, dtype=np.float64)
    nx = int(np.floor((maxs[0] - mins[0]) / xy_cell)) + 1
    ny = int(np.floor((maxs[1] - mins[1]) / xy_cell)) + 1
    nz = int(np.floor((maxs[2] - mins[2]) / z_cell)) + 1
    num = nx * ny * nz
    if num > max_voxels:
        raise ValueError(f"Voxel grid {nx} x {ny} x {nz} = {num:,} exceeds --max-voxels={max_voxels:,}; increase cell sizes")
    return mins, (nx, ny, nz)


def chunk_xyz(chunk):
    return (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))


def flat_voxels(x, y, z, mins, shape, xy_cell, z_cell):
    nx, ny, nz = shape
    ix = np.floor((x - mins[0]) / xy_cell).astype(np.int32)
    iy = np.floor((y - mins[1]) / xy_cell).astype(np.int32)
    iz = np.floor((z - mins[2]) / z_cell).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    np.clip(iz, 0, nz - 1, out=iz)
    return (ix.astype(np.int64) * ny + iy) * nz + iz


def derive_supported_components(counts, min_voxel_points, min_xy_support):
    """Only keep voxel evidence with at least K nearby XY cells at similar Z.

    3x3 XY neighbors allow +/- one Z bin to accommodate local slopes, but do
    not mistake 3D connectivity or this heuristic for proof of ground.
    """
    occupied = counts >= min_voxel_points
    similar_height = ndimage.maximum_filter(occupied, size=(1, 1, 3), mode="constant", cval=0)
    horizontal_count = ndimage.convolve(similar_height.astype(np.uint8),
                                        np.ones((3, 3, 1), dtype=np.uint8),
                                        mode="constant", cval=0)
    supported = occupied & (horizontal_count >= min_xy_support)
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, ncomponents = ndimage.label(supported, structure=structure)
    return labels, ncomponents, int(np.count_nonzero(occupied)), int(np.count_nonzero(supported))


def choose_component_seeds(labels, counts, mins, shape, initial_sensor,
                           xy_cell, z_cell, min_area, clearance,
                           max_stacking, min_3d_spacing, max_candidates, sensor_height):
    """Inspect 3D connected components and pick one deep interior cell each."""
    ncomp = int(labels.max())
    volumes = np.bincount(labels.ravel(), minlength=ncomp + 1)
    min_cells = int(np.ceil(min_area / (xy_cell * xy_cell)))
    slices = ndimage.find_objects(labels)
    proposals = []
    for comp_id in np.flatnonzero(volumes[1:] >= min_cells) + 1:
        roi = slices[int(comp_id) - 1]
        if roi is None:
            continue
        section = labels[roi] == comp_id
        xy = np.any(section, axis=2)
        nxy = int(np.count_nonzero(xy))
        area = nxy * xy_cell**2
        if area < min_area:
            continue
        stacking = float(volumes[comp_id] / nxy)
        if stacking > max_stacking:
            continue
        dist = ndimage.distance_transform_edt(np.pad(xy, 1))[1:-1, 1:-1] * xy_cell
        safe_i, safe_j = np.nonzero(xy & (dist >= clearance))
        if not len(safe_i):
            continue
        # Higher clearance is generally safer than perimeter origins.
        indices = np.argsort(dist[safe_i, safe_j])[::-1]
        selected = None
        for index in indices[:min(1500, len(indices))]:
            local_i = int(safe_i[index])
            local_j = int(safe_j[index])
            global_i = roi[0].start + local_i
            global_j = roi[1].start + local_j
            z_ids = np.flatnonzero(section[local_i, local_j])
            if not len(z_ids):
                continue
            # Prefer the most populated supported Z bin in this column.
            z_order = np.argsort(counts[roi][local_i, local_j, z_ids])[::-1]
            for zi_index in z_order:
                global_k = roi[2].start + int(z_ids[zi_index])
                center = np.array([mins[0] + (global_i + 0.5)*xy_cell,
                                   mins[1] + (global_j + 0.5)*xy_cell,
                                   mins[2] + (global_k + 0.5)*z_cell])
                # Compare ground positions, not sensor Z with ground Z.
                initial_ground = initial_sensor - np.array([0.0, 0.0, sensor_height])
                if np.linalg.norm(center - initial_ground) < min_3d_spacing:
                    continue
                selected = (global_i, global_j, global_k, center, float(dist[local_i, local_j]))
                break
            if selected is not None:
                break
        if selected is None:
            continue
        pi, pj, pk, center, cell_clearance = selected
        npoints = int(np.sum(counts[roi][section], dtype=np.int64))
        proposals.append(dict(component_id=int(comp_id), area_m2=round(area, 2),
                              voxel_count=int(volumes[comp_id]), point_count=npoints,
                              stacking=round(stacking, 3),
                              clearance_m=round(cell_clearance, 3),
                              voxel_i=pi, voxel_j=pj, voxel_k=pk,
                              approximate_x=float(center[0]),
                              approximate_y=float(center[1]),
                              approximate_z=float(center[2])))
    proposals.sort(key=lambda rec: (-rec["area_m2"], -rec["point_count"]))
    return proposals[:max_candidates], len(proposals)


def anchor_origins(input_las, proposals, mins, shape, xy_cell, z_cell, chunk_size):
    """Find a real classified-ground point inside every selected 3D voxel."""
    best_sq = [np.inf] * len(proposals)
    best_xyz = [None] * len(proposals)
    targets = [p["voxel_i"] * shape[1] * shape[2] + p["voxel_j"] * shape[2] + p["voxel_k"]
               for p in proposals]
    with laspy.open(input_las) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x, y, z = chunk_xyz(chunk)
            key = flat_voxels(x, y, z, mins, shape, xy_cell, z_cell)
            for r, target in enumerate(targets):
                matching = np.flatnonzero(key == target)
                if not len(matching):
                    continue
                dx = x[matching] - proposals[r]["approximate_x"]
                dy = y[matching] - proposals[r]["approximate_y"]
                dz = z[matching] - proposals[r]["approximate_z"]
                sq = dx*dx + dy*dy + dz*dz
                at = int(np.argmin(sq))
                if sq[at] < best_sq[r]:
                    best_sq[r] = float(sq[at])
                    k = int(matching[at])
                    best_xyz[r] = (float(x[k]), float(y[k]), float(z[k]))
    if any(x is None for x in best_xyz):
        raise RuntimeError("No supporting LAS point for a selected voxel")
    return best_xyz


def export_candidate_patches(input_las, output_dir, proposals, labels, mins,
                             shape, xy_cell, z_cell, chunk_size):
    """Export only the candidate's 3D-connected, horizontally supported voxels.

    Use these previews to REJECT roofs/objects; do not consider them ground truth.
    """
    with laspy.open(input_las) as reader, ExitStack() as stack:
        writers = [stack.enter_context(laspy.open(output_dir / f"candidate_patch_{i:02d}.las",
                                                  mode="w", header=reader.header))
                   for i in range(1, len(proposals) + 1)]
        saved = np.zeros(len(proposals), dtype=np.int64)
        ids = [p["component_id"] for p in proposals]
        for chunk in reader.chunk_iterator(chunk_size):
            x, y, z = chunk_xyz(chunk)
            key = flat_voxels(x, y, z, mins, shape, xy_cell, z_cell)
            lab = labels.ravel()[key]
            for i, comp_id in enumerate(ids):
                mask = lab == comp_id
                if np.any(mask):
                    subset = chunk[mask]
                    writers[i].write_points(subset)
                    saved[i] += len(subset)
    for i, n in enumerate(saved, start=1):
        print(f"  Preview candidate_patch_{i:02d}.las: {n:,} points", flush=True)


def discover(args):
    for path in (args.ground_las, args.ransac_las, args.initial_origins_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (args.xy_cell <= 0 or args.z_cell <= 0 or args.plane_distance <= 0
            or args.min_voxel_points < 1 or not 1 <= args.min_xy_support <= 9
            or args.min_area <= 0 or args.clearance < 0 or args.max_stacking < 1
            or args.max_candidates < 1 or args.export_top < 0
            or args.chunk_size < 1 or args.max_voxels < 1 or args.min_initial_distance < 0
            or args.sensor_height <= 0):
        raise ValueError("Invalid discover numeric argument")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    begin = perf_counter()
    a, slope = fit_ransac_plane(args.ransac_las, args.chunk_size)
    initial_sensor = read_sensor_xyz(read_csv(args.initial_origins_csv)[0])
    with laspy.open(args.ground_las) as reader:
        mins, shape = voxel_shape(reader.header, args.xy_cell, args.z_cell, args.max_voxels)
        n = int(reader.header.point_count)
    nv = int(np.prod(shape))
    counts = np.zeros(nv, dtype=np.uint32)
    outside = 0
    with laspy.open(args.ground_las) as reader:
        for chunk in reader.chunk_iterator(args.chunk_size):
            x, y, z = chunk_xyz(chunk)
            far_from_initial_plane = np.abs(z - (a[0]*x + a[1]*y + a[2])) >= args.plane_distance
            if np.any(far_from_initial_plane):
                flat = flat_voxels(x[far_from_initial_plane], y[far_from_initial_plane],
                                  z[far_from_initial_plane], mins, shape, args.xy_cell, args.z_cell)
                counts += np.bincount(flat, minlength=nv).astype(np.uint32)
                outside += int(np.count_nonzero(far_from_initial_plane))
    counts = counts.reshape(shape)
    labels, ncomps, occupied_count, supported_count = derive_supported_components(
        counts, args.min_voxel_points, args.min_xy_support)
    proposals, all_count = choose_component_seeds(
        labels, counts, mins, shape, initial_sensor,
        args.xy_cell, args.z_cell, args.min_area, args.clearance,
        args.max_stacking, args.min_initial_distance, args.max_candidates, args.sensor_height)
    print(f"Initial plane: z = {a[0]:.6f} x + {a[1]:.6f} y + {a[2]:.6f} (tilt {slope:.3f} deg)", flush=True)
    print(f"Origin #1 ground input: {n:,} points; outside plane +/-{args.plane_distance}m: {outside:,}", flush=True)
    print(f"3D voxel grid: {shape}; occupied={occupied_count:,}; horizontal-supported={supported_count:,}", flush=True)
    print(f"Candidate components: {ncomps:,}; passing geometric filters: {all_count:,}", flush=True)
    if not proposals:
        print("No eligible patches. Inspect parameters; do not assume no new ground exists.", flush=True)
        return
    anchors = anchor_origins(args.ground_las, proposals, mins, shape,
                             args.xy_cell, args.z_cell, args.chunk_size)
    records = []
    for i, (p, xyz) in enumerate(zip(proposals, anchors), 1):
        x, y, z = xyz
        records.append(dict(origin_id=i, component_id=p["component_id"],
                            area_m2_approx=p["area_m2"],
                            supported_voxels=p["voxel_count"],
                            supported_points_approx=p["point_count"],
                            vertical_stacking=p["stacking"],
                            clearance_m_approx=p["clearance_m"],
                            ground_x=round(x, 4), ground_y=round(y, 4), ground_z=round(z, 4),
                            sensor_x=round(x, 4), sensor_y=round(y, 4),
                            sensor_z=round(z + args.sensor_height, 4)))
    require_new_dir(args.output_dir)
    with (args.output_dir / "origins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    for name, cols in (("ground_origins.xyz", ("ground_x", "ground_y", "ground_z")),
                       ("sensor_origins.xyz", ("sensor_x", "sensor_y", "sensor_z"))):
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for r in records:
                f.write(" ".join(f"{r[col]:.4f}" for col in cols) + "\n")
    if args.export_top:
        export_candidate_patches(args.ground_las, args.output_dir,
                                 proposals[:args.export_top], labels,
                                 mins, shape, args.xy_cell, args.z_cell, args.chunk_size)
    for r in records:
        print(f"  #{r['origin_id']:02d}: area~{r['area_m2_approx']}m2, "
              f"ground=({r['ground_x']:.3f},{r['ground_y']:.3f},{r['ground_z']:.3f}), "
              f"clearance~{r['clearance_m_approx']}m", flush=True)
    print(f"Saved: {args.output_dir} ({perf_counter()-begin:.1f}s)", flush=True)


def patchwork_ground_indices(input_las, origin, matrix, chunk_size, min_range, zone0_sectors):
    import pypatchworkpp as pwp
    from multi_origin_patchworkpp import fill_patchwork_input

    fill_patchwork_input(input_las, matrix, origin, chunk_size)
    params = pwp.Parameters()
    params.min_range = min_range
    params.num_sectors_each_zone = [zone0_sectors, 32, 54, 32]
    params.enable_RNR = False
    params.verbose = False
    algo = pwp.patchworkpp(params)
    try:
        start = perf_counter()
        algo.estimateGround(matrix)
        seconds = perf_counter() - start
        indices = np.asarray(algo.getGroundIndices(), dtype=np.int64).ravel().copy()
        n_non = int(np.asarray(algo.getNongroundIndices()).size)
    finally:
        del algo
        gc.collect()
    if len(indices) and (indices.min() < 0 or indices.max() >= len(matrix)):
        raise RuntimeError("Out-of-range Patchwork++ ground indices")
    return indices, n_non, seconds


def run(args):
    for path in (args.full_las, args.initial_origins_csv, args.candidate_origins_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1 or args.min_range <= 0 or args.zone0_sectors < 1:
        raise ValueError("Invalid run numeric argument")
    existing_rows = read_csv(args.initial_origins_csv)
    candidates = read_csv(args.candidate_origins_csv)
    chosen = [r for r in candidates if str(r["origin_id"]) == str(args.origin_id)]
    if len(chosen) != 1:
        raise ValueError(f"--origin-id {args.origin_id} does not match exactly one candidate")
    old = read_sensor_xyz(existing_rows[0])
    new = read_sensor_xyz(chosen[0])
    require_new_dir(args.output_dir)
    with laspy.open(args.full_las) as reader:
        n = int(reader.header.point_count)
    scratch = args.output_dir / "_patchwork_input_f32.dat"
    points = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(n, 4))
    baseline = np.zeros(n, dtype=bool)
    new_mask = np.zeros(n, dtype=bool)
    t0 = perf_counter()
    try:
        for label, origin, mask in (("baseline_01", old, baseline),
                                    (f"expanded_{args.origin_id}", new, new_mask)):
            print(f"Run {label}: sensor={origin}", flush=True)
            indices, nonground, seconds = patchwork_ground_indices(
                args.full_las, origin, points, args.chunk_size,
                args.min_range, args.zone0_sectors)
            mask[indices] = True
            if int(np.count_nonzero(mask)) != len(indices):
                raise RuntimeError("Patchwork++ returned duplicate ground indices")
            if label == "baseline_01" and args.expected_baseline_count is not None:
                if len(indices) != args.expected_baseline_count:
                    raise RuntimeError(f"Baseline mismatch: {len(indices):,} vs expected {args.expected_baseline_count:,}")
            print(f"  Ground={len(indices):,} | NonGround={nonground:,} | segmentation={seconds:.2f}s", flush=True)
            del indices
            gc.collect()
        added = new_mask & ~baseline
        missed = baseline & ~new_mask
        overlap = new_mask & baseline
        metrics = dict(baseline_ground=int(np.count_nonzero(baseline)),
                       new_origin_ground=int(np.count_nonzero(new_mask)),
                       new_only_vs_baseline=int(np.count_nonzero(added)),
                       baseline_only=int(np.count_nonzero(missed)),
                       both_ground=int(np.count_nonzero(overlap)),
                       elapsed_seconds=round(perf_counter() - t0, 2))
        from multi_origin_patchworkpp import save_ground_subset
        save_ground_subset(args.full_las, args.output_dir / "expanded_origin_ground.las", new_mask, args.chunk_size)
        save_ground_subset(args.full_las, args.output_dir / "new_only_vs_origin01.las", added, args.chunk_size)
        with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics))
            w.writeheader()
            w.writerow(metrics)
        print("Comparison (no union): " + ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in metrics.items()), flush=True)
        print(f"Saved: {args.output_dir}", flush=True)
    finally:
        del points
        gc.collect()
        try:
            scratch.unlink(missing_ok=True)
        except PermissionError:
            print(f"Temporary scratch still in use, safe to delete after process exits: {scratch}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("discover", help="Discover NEW ground-patch candidates from origin #1 only")
    d.add_argument("ground_las", type=Path, help="outputs/multi_origin_08/origin_01_ground.las")
    d.add_argument("ransac_las", type=Path, help="outputs/ransac_plane_001.las")
    d.add_argument("initial_origins_csv", type=Path, help="outputs/virtual_origins_05/origins.csv")
    d.add_argument("output_dir", type=Path)
    d.add_argument("--xy-cell", type=float, default=0.5)
    d.add_argument("--z-cell", type=float, default=0.25)
    d.add_argument("--plane-distance", type=float, default=0.4,
                   help="Minimum vertical distance from initial RANSAC plane, first expansion only")
    d.add_argument("--min-voxel-points", type=int, default=12)
    d.add_argument("--min-xy-support", type=int, default=5, help="Min occupied neighbors in 3x3 XY window at similar Z")
    d.add_argument("--min-area", type=float, default=10.0)
    d.add_argument("--clearance", type=float, default=1.0)
    d.add_argument("--max-stacking", type=float, default=3.0,
                   help="Reject components with too many vertical layers per XY cell")
    d.add_argument("--min-initial-distance", type=float, default=4.0,
                   help="3D ground-to-ground distance from origin #1")
    d.add_argument("--sensor-height", type=float, default=1.723)
    d.add_argument("--max-candidates", type=int, default=8)
    d.add_argument("--export-top", type=int, default=3)
    d.add_argument("--max-voxels", type=int, default=8_000_000)
    d.add_argument("--chunk-size", type=int, default=1_000_000)
    d.set_defaults(func=discover)
    r = sub.add_parser("run", help="One independent Patchwork++ expansion and exact pointwise baseline difference")
    r.add_argument("full_las", type=Path, help="Full, original SKV1.las")
    r.add_argument("initial_origins_csv", type=Path)
    r.add_argument("candidate_origins_csv", type=Path)
    r.add_argument("output_dir", type=Path)
    r.add_argument("--origin-id", type=int, default=1, help="Reviewed candidate ID in origins.csv")
    r.add_argument("--min-range", type=float, default=0.5)
    r.add_argument("--zone0-sectors", type=int, default=16)
    r.add_argument("--chunk-size", type=int, default=1_000_000)
    r.add_argument("--expected-baseline-count", type=int, default=None,
                   help="Abort if original origin #1 does not reproduce this point count")
    r.set_defaults(func=run)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
