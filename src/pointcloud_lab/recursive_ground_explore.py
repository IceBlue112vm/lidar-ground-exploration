"""Experiment 04: recursive exploration of NEW Patchwork++ ground, without union.

Requirements: laspy, numpy, scipy, pypatchworkpp, and the existing
multi_origin_patchworkpp.py and recursive_ground_step1.py in this directory.

For each independent origin, record a point-index bitmask. For a new origin,
extract only Ground(child) \\ Ground(parent), discover locally supported 3-D
patches in that DIFFERENCE, and enqueue new origins on those patches.
No global Ground union is calculated, exported, or used to choose seeds.

The method deliberately favors exploration, not false-positive rejection.
Planar roofs, platforms, vegetation etc. may become seeds. Limits prevent
unbounded propagation; they DO NOT establish complete ground coverage.

First execution (example):
  uv run python -u src/pointcloud_lab/recursive_ground_explore.py \
      data/SKV1.las outputs/virtual_origins_05/origins.csv \
      outputs/recursive_04_full \
      --root-count 8 \
      --first-root-candidates outputs/recursive_04_candidates/origins.csv \
      --max-executions 64 --save-novel-every 4

Resume without rerunning completed origins:
  uv run python -u src/pointcloud_lab/recursive_ground_explore.py \
      data/SKV1.las outputs/virtual_origins_05/origins.csv \
      outputs/recursive_04_full --root-count 8 \
      --first-root-candidates outputs/recursive_04_candidates/origins.csv \
      --max-executions 64 --save-novel-every 4 --resume
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

from recursive_ground_step1 import (
    anchor_origins,
    chunk_xyz,
    derive_supported_components,
    flat_voxels,
    voxel_shape,
)


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
    from multi_origin_patchworkpp import fill_patchwork_input

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
        if out.exists() and any(out.iterdir()):
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
                from multi_origin_patchworkpp import save_ground_subset
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
                from multi_origin_patchworkpp import save_ground_subset
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


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("full_las", type=Path)
    p.add_argument("initial_origins_csv", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--root-count", type=int, default=8, help="Run the first N independent RANSAC-seeded origins (default 8)")
    p.add_argument("--first-root-candidates", type=Path, default=None,
                   help="Reuse step1 candidate CSV for the first root instead of re-discovering")
    p.add_argument("--max-executions", type=int, default=64,
                   help="Max Patchwork++ executions per invocation (resume to continue)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--expected-first-root-count", type=int, default=23028831)
    p.add_argument("--min-range", type=float, default=0.5)
    p.add_argument("--zone0-sectors", type=int, default=16)
    p.add_argument("--sensor-height", type=float, default=1.723)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--xy-cell", type=float, default=0.5)
    p.add_argument("--z-cell", type=float, default=0.25)
    p.add_argument("--min-voxel-points", type=int, default=12)
    p.add_argument("--min-xy-support", type=int, default=5)
    p.add_argument("--min-area", type=float, default=10.0)
    p.add_argument("--clearance", type=float, default=1.0)
    p.add_argument("--max-stacking", type=float, default=3.0)
    p.add_argument("--min-origin-distance", type=float, default=4.0, help="Minimum 3D ground-reference spacing")
    p.add_argument("--max-seeds-per-patch", type=int, default=3)
    p.add_argument("--max-candidates-per-result", type=int, default=12)
    p.add_argument("--max-voxels", type=int, default=8_000_000)
    p.add_argument("--save-novel-every", type=int, default=0,
                   help="Keep new-only LAS on every Nth origin; 0 saves none (avoid disk exhaustion)")
    p.add_argument("--save-ground-every", type=int, default=0,
                   help="Keep full independent Ground LAS on every Nth origin; 0 saves none")
    args = p.parse_args()
    if (args.root_count < 1 or args.max_executions < 1 or args.chunk_size < 1 or
            args.xy_cell <= 0 or args.z_cell <= 0 or args.min_voxel_points < 1 or
            not 1 <= args.min_xy_support <= 9 or args.min_area <= 0 or
            args.clearance < 0 or args.max_stacking < 1 or
            args.min_origin_distance <= 0 or args.max_seeds_per_patch < 1 or
            args.max_candidates_per_result < 1 or args.max_voxels < 1 or
            args.save_novel_every < 0 or args.save_ground_every < 0 or args.min_range <= 0 or args.sensor_height <= 0 or
            args.zone0_sectors < 1):
        p.error("Invalid numeric parameter")
    main(args)


if __name__ == "__main__":
    cli()
