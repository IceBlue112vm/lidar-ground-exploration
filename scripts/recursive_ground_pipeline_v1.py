"""Single-file, v1-frozen RANSAC -> origins -> recursive Patchwork++ pipeline.

This script is standalone in the project: no imports from other project .py files.
Dependencies: numpy, scipy, laspy, pypatchworkpp; open3d only when generating
RANSAC inliers automatically. Source LAS is always segmented independently at
*every origin*. No global Ground union is produced or used for exploration.

SKV1 exact previously generated origins and first-root candidates:
  uv run python -u scripts/recursive_ground_pipeline_v1.py \
      data/SKV1.las outputs/scene_skv1_v1_fixed \
      --initial-origins-csv outputs/virtual_origins_05/origins.csv \
      --first-root-candidates outputs/recursive_04_candidates/origins.csv \
      --expected-first-root-count 23028831 --max-total-executions 96

New scene from raw LAS (auto RANSAC bootstrap; open3d needed):
  uv run python -u scripts/recursive_ground_pipeline_v1.py \
      data/NEW_SCENE.las outputs/scene_new_v1 --max-total-executions 96

Run the SAME command with --resume to process the next batch. Default batch=32.
RANSAC fit and origin placement are only done on the first run; written into
output_dir/_bootstrap for reproducibility. If an external seed CSV is provided,
that preprocessing step is skipped.

WARNING: This is a research prototype. The heuristic can follow vegetation,
roofs, and other non-ground surfaces. 96 is a *budget*, NOT a convergence or
accuracy guarantee. 128-run v1 output is NOT modified. To exactly reproduce
old SKV1 seed ordering, pass BOTH CSVs from the earlier experiments.
"""


import argparse
import csv
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from scipy import ndimage



# Local copies of the unchanged v1 helper logic.

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



def fill_patchwork_input(input_las: Path, points: np.memmap, origin: np.ndarray, chunk_size: int):
    """Fill Nx4 float32 memmap in original LAS order, translated to origin."""
    start = 0
    with laspy.open(input_las) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            end = start + len(chunk)
            points[start:end, 0] = np.asarray(chunk.x) - origin[0]
            points[start:end, 1] = np.asarray(chunk.y) - origin[1]
            points[start:end, 2] = np.asarray(chunk.z) - origin[2]
            points[start:end, 3] = np.asarray(chunk.intensity, dtype=np.float32)
            start = end
    if start != len(points):
        raise RuntimeError(f"LAS row count mismatch: expected {len(points):,}, got {start:,}")
    points.flush()


def save_ground_subset(input_las: Path, path: Path, selected: np.ndarray, chunk_size: int):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    saved = 0
    offset = 0
    with laspy.open(input_las) as reader:
        with laspy.open(path, mode="w", header=reader.header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                count = len(chunk)
                mask = selected[offset:offset + count]
                if np.any(mask):
                    subset = chunk[mask]
                    writer.write_points(subset)
                    saved += len(subset)
                offset += count
    if offset != len(selected) or saved != int(np.count_nonzero(selected)):
        raise RuntimeError("LAS subset export did not preserve expected point count")
    print(f"Saved: {path} ({saved:,} points)", flush=True)


from types import SimpleNamespace

ORIGIN_FIELDS = [
    "origin_id", "component_rank", "component_id", "component_area_m2_approx",
    "component_inlier_points", "ground_x", "ground_y", "ground_z",
    "sensor_x", "sensor_y", "sensor_z", "clearance_m_approx",
]

def generate_initial_origins(args):
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
        writer = csv.DictWriter(f, fieldnames=ORIGIN_FIELDS)
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


SCHEMA_VERSION = 1
FIELDS = [
    "id", "parent_id", "depth", "kind", "source_component", "origin_x", "origin_y",
    "origin_z", "ground_points", "new_vs_parent", "candidates_enqueued",
    "segmentation_seconds", "elapsed_seconds", "ground_mask_file",
]


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_reports(out, state, sensor_height):
    write_json_atomic(out / "state.json", state)
    with (out / "lineage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in state["processed"]:
            writer.writerow({key: item.get(key, "") for key in FIELDS})
    with (out / "origins_executed.xyz").open("w", encoding="utf-8") as f:
        for item in state["processed"]:
            f.write(f"{item['origin_x']:.4f} {item['origin_y']:.4f} {item['origin_z']:.4f}\n")
    with (out / "origins_pending.xyz").open("w", encoding="utf-8") as f:
        for item in state["pending"]:
            xyz = item["sensor"]
            f.write(f"{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")


def read_origins(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No origins in {path}")
    origins = []
    for row in rows:
        xyz = [float(row[k]) for k in ("sensor_x", "sensor_y", "sensor_z")]
        if not np.isfinite(xyz).all():
            raise ValueError(f"Invalid coordinates in {path}: {row}")
        origins.append(xyz)
    return origins


def save_mask(path, mask):
    """~8.5 MiB on disk per 71M-point classification (no global union)."""
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, np.packbits(mask))
    os.replace(temporary, path)


def load_mask(path, n):
    bits = np.load(path, allow_pickle=False)
    if len(bits) != (n + 7) // 8:
        raise ValueError(f"Saved mask length mismatch: {path}")
    return np.unpackbits(bits, count=n).astype(bool, copy=False)


def ground_mask(full_las, sensor, matrix, chunk_size, min_range, zone0_sectors):
    """Run PWP on ALL original points; return independent boolean classification."""
    import pypatchworkpp as pwp

    n = len(matrix)
    t0 = perf_counter()
    fill_patchwork_input(full_las, matrix, np.asarray(sensor), chunk_size)
    params = pwp.Parameters()
    params.min_range = min_range
    params.num_sectors_each_zone = [zone0_sectors, 32, 54, 32]
    params.enable_RNR = False
    params.verbose = False
    algo = pwp.patchworkpp(params)
    try:
        seg_start = perf_counter()
        algo.estimateGround(matrix)
        seg_seconds = perf_counter() - seg_start
        indices = np.asarray(algo.getGroundIndices(), dtype=np.int64).ravel().copy()
        nonground = int(np.asarray(algo.getNongroundIndices()).size)
    finally:
        del algo
        gc.collect()
    if len(indices) and (indices.min() < 0 or indices.max() >= n):
        raise RuntimeError("Patchwork++ returned out-of-range indices")
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    if int(np.count_nonzero(mask)) != len(indices):
        raise RuntimeError("Duplicate Ground indices from Patchwork++")
    if n - len(indices) - nonground != 0:
        raise RuntimeError("Unexpected unclassified point count in Patchwork++")
    return mask, round(seg_seconds, 3), round(perf_counter() - t0, 3)


def far_enough(ground_xyz, known_ground_xyz, minimum_distance):
    if not known_ground_xyz:
        return True
    p = np.asarray(ground_xyz, dtype=np.float64)
    q = np.asarray(known_ground_xyz, dtype=np.float64)
    return bool(np.min(np.linalg.norm(q - p, axis=1)) >= minimum_distance)


def discover_seeds(ground_las, known_ground_xyz, args):
    """3D voxel patches in ONE per-origin classification/difference LAS.

    Unlike step1, this never references the initial RANSAC plane. Seeds are
    anchored to actual points in a candidate voxel. 3D rather than XY-only
    separation distinguishes stacked floors where geometry permits.
    """
    with laspy.open(ground_las) as reader:
        n = int(reader.header.point_count)
        if n == 0:
            return [], {"components": 0, "supported_voxels": 0, "input_points": 0}
        mins, shape = voxel_shape(reader.header, args.xy_cell, args.z_cell, args.max_voxels)
    nv = int(np.prod(shape))
    counts = np.zeros(nv, dtype=np.uint32)
    with laspy.open(ground_las) as reader:
        for chunk in reader.chunk_iterator(args.chunk_size):
            x, y, z = chunk_xyz(chunk)
            flat = flat_voxels(x, y, z, mins, shape, args.xy_cell, args.z_cell)
            counts += np.bincount(flat, minlength=nv).astype(np.uint32)
    counts = counts.reshape(shape)
    labels, ncomps, occupied, supported = derive_supported_components(
        counts, args.min_voxel_points, args.min_xy_support)
    volumes = np.bincount(labels.ravel(), minlength=ncomps + 1)
    rois = ndimage.find_objects(labels)
    known = [tuple(xyz) for xyz in known_ground_xyz]
    proposals = []
    min_cells = int(np.ceil(args.min_area / (args.xy_cell**2)))
    # Favor larger, spatially coherent components, but allow multiple origins
    # within a large component instead of just one per component.
    ranked = np.flatnonzero(volumes[1:] >= min_cells) + 1
    ranked = sorted(ranked, key=lambda cid: int(volumes[cid]), reverse=True)
    for cid in ranked:
        if len(proposals) >= args.max_candidates_per_result:
            break
        roi = rois[int(cid) - 1]
        if roi is None:
            continue
        section = labels[roi] == cid
        xy = np.any(section, axis=2)
        xy_cells = int(np.count_nonzero(xy))
        area = xy_cells * args.xy_cell**2
        if area < args.min_area:
            continue
        vertical_stacking = float(volumes[cid] / max(1, xy_cells))
        if vertical_stacking > args.max_stacking:
            continue
        distance = ndimage.distance_transform_edt(np.pad(xy, 1))[1:-1, 1:-1] * args.xy_cell
        safe_i, safe_j = np.nonzero(xy & (distance >= args.clearance))
        if len(safe_i) == 0:
            continue
        # For each safe XY grid position, choose its densest supported Z bin.
        local_counts = counts[roi]
        zflat = np.full(len(safe_i), -1, dtype=np.int32)
        for p, (li, lj) in enumerate(zip(safe_i, safe_j)):
            zids = np.flatnonzero(section[li, lj])
            if len(zids):
                zflat[p] = int(zids[np.argmax(local_counts[li, lj, zids])])
        valid = zflat >= 0
        if not np.any(valid):
            continue
        si, sj, sk = safe_i[valid], safe_j[valid], zflat[valid]
        centers = np.column_stack((
            mins[0] + (roi[0].start + si + 0.5) * args.xy_cell,
            mins[1] + (roi[1].start + sj + 0.5) * args.xy_cell,
            mins[2] + (roi[2].start + sk + 0.5) * args.z_cell,
        ))
        clearance = distance[si, sj]
        for _ in range(args.max_seeds_per_patch):
            if len(proposals) >= args.max_candidates_per_result or len(centers) == 0:
                break
            if known:
                # Sequential nearest distance avoids an N_candidates x N_origins
                # temporary matrix and permits many recursively visited origins.
                dsq = np.full(len(centers), np.inf, dtype=np.float64)
                for xyz in known:
                    diff = centers - np.asarray(xyz)
                    dsq = np.minimum(dsq, np.einsum("ij,ij->i", diff, diff))
                valid = dsq >= args.min_origin_distance**2
                if not np.any(valid):
                    break
                # Encourage exploration far from existing visited/queued origins.
                score = np.sqrt(dsq) + 0.02 * clearance
            else:
                valid = np.ones(len(centers), dtype=bool)
                score = clearance.copy()
            score[~valid] = -np.inf
            best = int(np.argmax(score))
            if not np.isfinite(score[best]):
                break
            center = centers[best]
            proposal = {
                "component_id": int(cid),
                "area_m2_approx": round(area, 2),
                "clearance_m_approx": round(float(clearance[best]), 3),
                "voxel_i": int(roi[0].start + si[best]),
                "voxel_j": int(roi[1].start + sj[best]),
                "voxel_k": int(roi[2].start + sk[best]),
                "approximate_x": float(center[0]),
                "approximate_y": float(center[1]),
                "approximate_z": float(center[2]),
            }
            proposals.append(proposal)
            known.append(tuple(center))
    if not proposals:
        return [], {"components": int(ncomps), "supported_voxels": int(supported), "input_points": n}
    actual_xyz = anchor_origins(ground_las, proposals, mins, shape,
                                args.xy_cell, args.z_cell, args.chunk_size)
    result = []
    for rec, xyz in zip(proposals, actual_xyz):
        # Anchor is an observed point in a supported voxel. Sensor Z is added ONCE.
        result.append({
            "ground": [round(float(v), 4) for v in xyz],
            "sensor": [round(float(xyz[0]), 4), round(float(xyz[1]), 4),
                       round(float(xyz[2] + args.sensor_height), 4)],
            "component_id": rec["component_id"],
            "area_m2_approx": rec["area_m2_approx"],
            "clearance_m_approx": rec["clearance_m_approx"],
        })
    return result, {"components": int(ncomps), "supported_voxels": int(supported), "input_points": n}


def enqueue(state, proposed, parent_id, parent_depth, kind, args):
    all_known = []
    for p in state["processed"]:
        all_known.append((p["origin_x"], p["origin_y"], p["origin_z"] - args.sensor_height))
    for p in state["pending"]:
        all_known.append(tuple(np.asarray(p["sensor"]) - [0, 0, args.sensor_height]))
    added = 0
    for candidate in proposed:
        ground = candidate["ground"] if "ground" in candidate else [
            candidate["sensor"][0], candidate["sensor"][1],
            candidate["sensor"][2] - args.sensor_height,
        ]
        if not far_enough(ground, all_known, args.min_origin_distance):
            continue
        state["pending"].append({
            "parent_id": parent_id,
            "depth": parent_depth + 1,
            "kind": kind,
            "source_component": candidate.get("component_id", ""),
            "sensor": list(candidate["sensor"]),
        })
        all_known.append(tuple(ground))
        added += 1
    return added


def parse_first_root_candidates(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [{"sensor": [float(row[k]) for k in ("sensor_x", "sensor_y", "sensor_z")],
             "component_id": row.get("component_id", ""),
             "ground": [float(row[k]) for k in ("ground_x", "ground_y", "ground_z")]}
            for row in rows]


def config_of(args, n):
    return {
        "full_las": str(args.full_las.resolve()),
        "initial_origins_csv": str(args.initial_origins_csv.resolve()),
        "first_root_candidates": str(args.first_root_candidates.resolve()) if args.first_root_candidates else None,
        "root_count": args.root_count,
        "point_count": n,
        "xy_cell": args.xy_cell, "z_cell": args.z_cell,
        "min_voxel_points": args.min_voxel_points, "min_xy_support": args.min_xy_support,
        "min_area": args.min_area, "clearance": args.clearance,
        "max_stacking": args.max_stacking, "max_voxels": args.max_voxels,
        "min_origin_distance": args.min_origin_distance,
        "max_candidates_per_result": args.max_candidates_per_result,
        "max_seeds_per_patch": args.max_seeds_per_patch,
        "sensor_height": args.sensor_height, "min_range": args.min_range,
        "zone0_sectors": args.zone0_sectors,
    }


def main(args):
    if not args.full_las.is_file() or not args.initial_origins_csv.is_file():
        raise FileNotFoundError("Full LAS or initial origins CSV not found")
    if args.first_root_candidates and not args.first_root_candidates.is_file():
        raise FileNotFoundError(args.first_root_candidates)
    with laspy.open(args.full_las) as reader:
        n = int(reader.header.point_count)
    origins = read_origins(args.initial_origins_csv)
    if args.root_count > len(origins):
        raise ValueError(f"Asked for {args.root_count} roots, only {len(origins)} available")
    out = args.output_dir
    config = config_of(args, n)
    if args.resume:
        state_file = out / "state.json"
        if not state_file.is_file():
            raise FileNotFoundError(f"No previous state to resume: {state_file}")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["version"] != SCHEMA_VERSION or state["config"] != config:
            raise ValueError("Resume settings differ from saved experiment. Use exact discovery/segmentation args")
    else:
        if out.exists() and any(p.name != "_bootstrap" for p in out.iterdir()):
            raise FileExistsError(f"Output directory must be empty: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(exist_ok=True)
        (out / "samples").mkdir(exist_ok=True)
        state = {"version": SCHEMA_VERSION, "config": config, "next_id": 0,
                 "processed": [], "pending": [
                     {"parent_id": None, "depth": 0, "kind": "initial_ransac_origin",
                      "source_component": "", "sensor": xyz} for xyz in origins[:args.root_count]
                 ]}
        save_reports(out, state, args.sensor_height)

    scratch = out / "_patchwork_input_f32.dat"
    scratch.unlink(missing_ok=True)
    print(f"Full LAS: {n:,}; initial roots: {args.root_count}; pending: {len(state['pending'])}", flush=True)
    print(f"Input memmap: {n * 16 / (1024**3):.2f} GiB; NO ground union", flush=True)
    matrix = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(n, 4))
    executions_this_run = 0
    start = perf_counter()
    try:
        while state["pending"] and executions_this_run < args.max_executions:
            item = state["pending"][0]
            parent_id = item["parent_id"]
            origin = item["sensor"]
            # Candidates may have become redundant after earlier branches ran.
            if parent_id is not None:
                old_ground = [(p["origin_x"], p["origin_y"], p["origin_z"] - args.sensor_height)
                              for p in state["processed"]]
                point_ground = [origin[0], origin[1], origin[2] - args.sensor_height]
                if not far_enough(point_ground, old_ground, args.min_origin_distance):
                    print(f"Skip queued origin near an already executed origin: {origin}", flush=True)
                    state["pending"].pop(0)
                    save_reports(out, state, args.sensor_height)
                    continue
            current_id = state["next_id"]
            t0 = perf_counter()
            print(f"\n[{current_id:04d}] depth={item['depth']}, parent={parent_id}, "
                  f"kind={item['kind']}, sensor={np.round(origin,4)}", flush=True)
            mask, seg_seconds, _ = ground_mask(args.full_las, origin, matrix,
                                                args.chunk_size, args.min_range, args.zone0_sectors)
            count = int(np.count_nonzero(mask))
            if current_id == 0 and args.expected_first_root_count is not None:
                if count != args.expected_first_root_count:
                    raise RuntimeError(f"First root count changed: {count:,} vs {args.expected_first_root_count:,}")
            parent_mask = None
            if parent_id is None:
                novel_mask = mask  # root classifies independently; no parent exists
            else:
                parent_row = next((p for p in state["processed"] if p["id"] == parent_id), None)
                if parent_row is None:
                    raise RuntimeError(f"Cannot locate parent mask for {parent_id}")
                parent_mask = load_mask(out / parent_row["ground_mask_file"], n)
                novel_mask = mask & ~parent_mask  # child-only Ground; NOT a union
            new_points = int(np.count_nonzero(novel_mask))
            print(f"  Ground={count:,}; new_vs_parent={new_points:,}; segmentation={seg_seconds:.2f}s", flush=True)
            if args.save_ground_every and current_id % args.save_ground_every == 0:
                snapshot = out / "samples" / f"ground_{current_id:04d}.las"
                # A previous interrupted attempt may already have saved this
                # full LAS. Avoid a needless overwrite when resuming.
                if not snapshot.exists():
                    save_ground_subset(args.full_las, snapshot, mask, args.chunk_size)
            mask_path = out / "masks" / f"ground_{current_id:04d}.npy"
            save_mask(mask_path, mask)
            del mask, parent_mask
            gc.collect()
            # For the first root, reuse already verified step1 candidates.
            if current_id == 0 and args.first_root_candidates:
                candidate_list = parse_first_root_candidates(args.first_root_candidates)
                details = {"components": "precomputed", "input_points": new_points}
            elif new_points:
                temporary_las = out / f"_novel_{current_id:04d}.las"
                if temporary_las.exists():
                    temporary_las.unlink()
                save_ground_subset(args.full_las, temporary_las, novel_mask, args.chunk_size)
                known = [(p["origin_x"], p["origin_y"], p["origin_z"] - args.sensor_height)
                         for p in state["processed"]]
                known += [[v[0], v[1], v[2] - args.sensor_height] for v in origins[:args.root_count]]
                known += [[v["sensor"][0], v["sensor"][1], v["sensor"][2] - args.sensor_height]
                          for v in state["pending"]]
                try:
                    candidate_list, details = discover_seeds(temporary_las, known, args)
                    if args.save_novel_every and current_id % args.save_novel_every == 0:
                        os.replace(temporary_las, out / "samples" / f"novel_{current_id:04d}.las")
                    else:
                        temporary_las.unlink()
                except Exception:
                    print(f"  Retained temporary novel LAS for diagnosis: {temporary_las}", flush=True)
                    raise
            else:
                candidate_list, details = [], {"components": 0, "input_points": 0}
            del novel_mask
            gc.collect()
            accepted = enqueue(state, candidate_list, current_id, item["depth"],
                               "recursive_candidate", args)
            # NOTE: queued root origin stays at head until *after* this enqueue.
            state["pending"].pop(0)
            record = {
                "id": current_id, "parent_id": parent_id, "depth": item["depth"],
                "kind": item["kind"], "source_component": item.get("source_component", ""),
                "origin_x": round(float(origin[0]), 4), "origin_y": round(float(origin[1]), 4),
                "origin_z": round(float(origin[2]), 4), "ground_points": count,
                "new_vs_parent": new_points if parent_id is not None else "",
                "candidates_enqueued": accepted, "segmentation_seconds": seg_seconds,
                "elapsed_seconds": round(perf_counter() - t0, 3),
                "ground_mask_file": str(mask_path.relative_to(out)).replace("\\", "/"),
            }
            state["processed"].append(record)
            state["next_id"] += 1
            save_reports(out, state, args.sensor_height)
            if parent_id is None:
                print(f"  Root: independent Ground; queued={accepted}; components={details['components']}", flush=True)
            else:
                print(f"  New-only: {new_points:,}; queued={accepted}; components={details['components']}; "
                      f"pending={len(state['pending'])}", flush=True)
            executions_this_run += 1
        cause = ("candidate queue exhausted" if not state["pending"]
                 else f"safety cap reached ({args.max_executions} executions this invocation)")
        print(f"\nStopped: {cause}. Executed total={len(state['processed'])}, "
              f"pending={len(state['pending'])}, elapsed={perf_counter()-start:.1f}s", flush=True)
        print(f"Files: {out / 'lineage.csv'}, {out / 'state.json'}, "
              f"{out / 'origins_executed.xyz'}", flush=True)
        print("No union has been computed. 'queue exhausted' is NOT proof all real ground was found.", flush=True)
    finally:
        del matrix
        gc.collect()
        try:
            scratch.unlink(missing_ok=True)
        except PermissionError:
            print(f"Warning: delete scratch manually after exit: {scratch}", flush=True)




def detect_initial_ransac_inliers(full_las: Path, inlier_las: Path,
                                  iterations: int, threshold: float, seed: int,
                                  chunk_size: int, max_tilt_deg: float):
    """Full-resolution Open3D plane RANSAC, copied as an *independent*
    bootstrapping step. Results may not be byte-identical to the old
    ransac_full.py because Open3D implementations and randomization can vary.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Install open3d or supply --bootstrap-ransac-inliers / --initial-origins-csv") from exc
    with laspy.open(full_las) as reader:
        n = int(reader.header.point_count)
    if n == 0:
        raise ValueError(f"Empty full LAS: {full_las}")
    inlier_las.parent.mkdir(parents=True, exist_ok=True)
    scratch = inlier_las.with_name("_ransac_xyz_float64.dat")
    scratch.unlink(missing_ok=True)
    xyz = np.memmap(scratch, dtype=np.float64, mode="w+", shape=(n, 3))
    print(f"Bootstrap RANSAC: full resolution, {n:,} points, {iterations} iters, {threshold:g}m threshold",flush=True)
    try:
        k = 0
        with laspy.open(full_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                m = len(chunk)
                x,y,z = chunk_xyz(chunk)
                xyz[k:k+m,0] = x
                xyz[k:k+m,1] = y
                xyz[k:k+m,2] = z
                k += m
        if k != n:
            raise RuntimeError(f"Incomplete original LAS read: {k}/{n}")
        xyz.flush()
        if hasattr(o3d.utility, "random"):
            o3d.utility.random.seed(seed)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz))
        coefficients, selected = cloud.segment_plane(distance_threshold=threshold,
                                                       ransac_n=3,
                                                       num_iterations=iterations)
        del cloud
        a,b,c,d = [float(v) for v in coefficients]
        tilt = float(np.degrees(np.arccos(np.clip(abs(c)/np.linalg.norm([a,b,c]),0,1))))
        print(f"Bootstrap plane: {a:.7f}x + {b:.7f}y + {c:.7f}z + {d:.7f} = 0; tilt={tilt:.2f}deg; inliers={len(selected):,}",flush=True)
        if tilt > max_tilt_deg:
            raise RuntimeError(f"RANSAC selected plane tilt {tilt:.1f} > {max_tilt_deg:.1f} degrees. Inspect scene or supply verified seed origins.")
        chosen = np.zeros(n,dtype=bool)
        idx = np.asarray(selected,dtype=np.int64)
        if np.any((idx<0)|(idx>=n)):
            raise RuntimeError("RANSAC returned invalid row indices")
        chosen[idx] = True
        if int(chosen.sum()) != len(idx):
            raise RuntimeError("RANSAC returned duplicate inlier indices")
        save_ground_subset(full_las,inlier_las,chosen,chunk_size)
        del chosen, idx
    finally:
        del xyz
        gc.collect()
        try: scratch.unlink(missing_ok=True)
        except PermissionError: print(f"Warning: remove RANSAC scratch manually: {scratch}",flush=True)


def prepare_initial_origins(args):
    """Return path of origins.csv, generating RANSAC/placement only if needed."""
    if args.initial_origins_csv is not None:
        return args.initial_origins_csv
    bootstrap = args.output_dir / "_bootstrap"
    origin_file = bootstrap / "origins" / "origins.csv"
    if args.resume:
        if not origin_file.is_file():
            raise FileNotFoundError(f"Missing initial origins on resume: {origin_file}")
        return origin_file
    if args.output_dir.exists() and any(p.name != "_bootstrap" for p in args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty for new run: {args.output_dir}")
    if origin_file.is_file():
        print(f"Reuse completed initial origin placement: {origin_file}",flush=True)
        return origin_file
    bootstrap.mkdir(parents=True,exist_ok=True)
    inliers = args.bootstrap_ransac_inliers
    if inliers is None:
        inliers = bootstrap / "ransac_initial_inliers.las"
        detect_initial_ransac_inliers(args.full_las, inliers, args.ransac_iterations,
                                      args.ransac_threshold,args.ransac_seed,
                                      args.chunk_size,args.ransac_max_tilt)
    if not inliers.is_file():
        raise FileNotFoundError(f"RANSAC inlier LAS not found: {inliers}")
    origin_args=SimpleNamespace(
        input=inliers,
        output_dir=bootstrap/"origins",
        cell_size=args.initial_cell_size,
        min_points_per_cell=args.initial_min_points_per_cell,
        min_area=args.initial_min_area,
        max_components=args.initial_max_components,
        max_origins=args.root_count,
        clearance=args.initial_clearance,
        min_spacing=args.initial_min_spacing,
        sensor_height=args.sensor_height,
        max_grid_cells=args.initial_max_grid_cells,
        connectivity=8,
    )
    generate_initial_origins(origin_args)
    return origin_file


def cli():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("full_las",type=Path)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--initial-origins-csv",type=Path,default=None,
                   help="Use verified existing initial origins, skipping RANSAC and initial placement")
    p.add_argument("--bootstrap-ransac-inliers",type=Path,default=None,
                   help="Use existing RANSAC inlier LAS; automatically generate origins")
    p.add_argument("--first-root-candidates",type=Path,default=None,
                   help="Use earlier step1 candidate CSV for first root; optional")
    p.add_argument("--root-count",type=int,default=8)
    p.add_argument("--max-total-executions",type=int,default=96,
                   help="Hard TOTAL cap across resume calls (default 96)")
    p.add_argument("--batch-executions",type=int,default=32,
                   help="Maximum additional executions on this invocation (default 32)")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--bootstrap-only",action="store_true",
                   help="Run initial RANSAC/origin placement only, to inspect origins in CloudCompare")
    p.add_argument("--expected-first-root-count",type=int,default=None,
                   help="Optional SKV1 reproducibility guard; not portable to other scenes")
    # v1 segmentation/discovery defaults, preserved
    p.add_argument("--min-range",type=float,default=0.5)
    p.add_argument("--zone0-sectors",type=int,default=16)
    p.add_argument("--sensor-height",type=float,default=1.723)
    p.add_argument("--chunk-size",type=int,default=1_000_000)
    p.add_argument("--xy-cell",type=float,default=0.5)
    p.add_argument("--z-cell",type=float,default=0.25)
    p.add_argument("--min-voxel-points",type=int,default=12)
    p.add_argument("--min-xy-support",type=int,default=5)
    p.add_argument("--min-area",type=float,default=10.0)
    p.add_argument("--clearance",type=float,default=1.0)
    p.add_argument("--max-stacking",type=float,default=3.0)
    p.add_argument("--min-origin-distance",type=float,default=4.0)
    p.add_argument("--max-seeds-per-patch",type=int,default=3)
    p.add_argument("--max-candidates-per-result",type=int,default=12)
    p.add_argument("--max-voxels",type=int,default=8_000_000)
    p.add_argument("--save-novel-every",type=int,default=8)
    p.add_argument("--save-ground-every",type=int,default=0)
    # original initial-RANSAC and origins placement defaults
    p.add_argument("--ransac-iterations",type=int,default=1000)
    p.add_argument("--ransac-threshold",type=float,default=0.05)
    p.add_argument("--ransac-seed",type=int,default=42)
    p.add_argument("--ransac-max-tilt",type=float,default=25.0)
    p.add_argument("--initial-cell-size",type=float,default=0.5)
    p.add_argument("--initial-min-points-per-cell",type=int,default=3)
    p.add_argument("--initial-min-area",type=float,default=10.0)
    p.add_argument("--initial-max-components",type=int,default=4)
    p.add_argument("--initial-clearance",type=float,default=1.0)
    p.add_argument("--initial-min-spacing",type=float,default=10.0)
    p.add_argument("--initial-max-grid-cells",type=int,default=8_000_000)
    args=p.parse_args()
    if (args.root_count<1 or args.max_total_executions<1 or args.batch_executions<1 or
        args.chunk_size<1 or args.xy_cell<=0 or args.z_cell<=0 or args.min_voxel_points<1 or
        not 1<=args.min_xy_support<=9 or args.min_area<=0 or args.clearance<0 or
        args.max_stacking<1 or args.min_origin_distance<=0 or args.max_seeds_per_patch<1 or
        args.max_candidates_per_result<1 or args.max_voxels<1 or
        args.save_novel_every<0 or args.save_ground_every<0 or args.min_range<=0 or
        args.sensor_height<=0 or args.zone0_sectors<1 or args.ransac_iterations<1 or
        args.ransac_threshold<=0 or args.initial_cell_size<=0 or
        args.initial_min_points_per_cell<1 or args.initial_min_area<=0 or
        args.initial_max_components<1 or args.initial_clearance<0 or
        args.initial_min_spacing<=0 or args.initial_max_grid_cells<1):
        p.error("Invalid numeric parameter")
    if not args.full_las.is_file():
        raise FileNotFoundError(args.full_las)
    initial_origins_csv=prepare_initial_origins(args)
    args.initial_origins_csv=initial_origins_csv
    if args.bootstrap_only:
        print(f"Bootstrap complete; inspect origins in CloudCompare: {initial_origins_csv}",flush=True)
        return
    available=read_origins(initial_origins_csv)
    if args.root_count>len(available):
        print(f"Only {len(available)} initial origins available; reducing root-count {args.root_count} -> {len(available)}",flush=True)
        args.root_count=len(available)
    if args.resume:
        state_file=args.output_dir/"state.json"
        if not state_file.is_file():
            raise FileNotFoundError(f"Missing state for --resume: {state_file}")
        state=json.loads(state_file.read_text(encoding="utf-8"))
        done=len(state["processed"])
        if not state.get("pending"):
            print(f"Already complete; queue exhausted at {done} origins.",flush=True)
            return
    else:
        done=0
    remaining=args.max_total_executions-done
    if remaining<=0:
        print(f"Reached total cap: {done}/{args.max_total_executions}; nothing to execute.",flush=True)
        return
    args.max_executions=min(args.batch_executions,remaining)
    print(f"Pipeline v1 frozen: already={done}, this_batch<= {args.max_executions}, total_cap={args.max_total_executions}",flush=True)
    main(args)


if __name__ == "__main__":
    cli()
