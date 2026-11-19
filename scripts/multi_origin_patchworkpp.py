"""Run Patchwork++ on the *full* LAS once per generated virtual origin.

Input origin CSV is produced by generate_virtual_origins.py and includes
sensor_x, sensor_y, sensor_z. Each row is an actual *sensor* XYZ already:
DO NOT add sensor_height again.

This script deliberately uses a disk-backed float32 point matrix rather than
building a large in-memory LAS xyz array. A point's ground classification is
tracked by its original zero-based row index, so union contains no duplicates.

Run a one-origin smoke test before processing all 8 origins on ~71M points.
"""

import argparse
import csv
import gc
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
import pypatchworkpp as pwp


METRIC_COLUMNS = [
    "origin_id", "origin_x", "origin_y", "origin_z", "ground_points",
    "nonground_points", "unclassified_points", "new_ground_points",
    "cumulative_ground_points", "cumulative_ground_pct", "segmentation_seconds",
]


def read_origins(path: Path, limit: int):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No origins found in {path}")
    required = {"origin_id", "sensor_x", "sensor_y", "sensor_z"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Origin CSV must contain columns: {', '.join(sorted(required))}")
    origins = []
    for row in rows[:limit]:
        xyz = np.array([float(row[k]) for k in ("sensor_x", "sensor_y", "sensor_z")], dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"Non-finite origin coordinates in CSV: {row}")
        origins.append((str(row["origin_id"]), xyz))
    return origins


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_las", type=Path, help="Full original LAS (not RANSAC inliers)")
    parser.add_argument("origins_csv", type=Path, help="Generated origins.csv")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit-origins", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--min-range", type=float, default=0.5)
    parser.add_argument("--zone0-sectors", type=int, default=16)
    parser.add_argument("--skip-las-output", action="store_true", help="Only save per-origin CSV metrics; no LAS exports")
    args = parser.parse_args()

    if (args.limit_origins < 1 or args.chunk_size < 1 or args.min_range <= 0
            or args.zone0_sectors < 1):
        parser.error("Invalid numeric argument")
    if not args.input_las.is_file() or not args.origins_csv.is_file():
        parser.error("Input LAS or origins CSV does not exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    origins = read_origins(args.origins_csv, args.limit_origins)
    with laspy.open(args.input_las) as reader:
        n = int(reader.header.point_count)
    print(f"Full LAS: {n:,} points; origins: {len(origins)}", flush=True)
    print(f"Disk-backed input matrix: {n * 16 / 2**30:.2f} GiB", flush=True)

    # Flags refer to the original LAS point order, not local Patchwork coordinates.
    union = np.zeros(n, dtype=np.bool_)
    first_origin_mask = None
    total_start = perf_counter()
    scratch = args.output_dir / "_patchwork_input_f32.dat"
    points = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(n, 4))

    try:
        with (args.output_dir / "origin_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            writer.writeheader()

            for iteration, (origin_id, origin) in enumerate(origins, start=1):
                print(f"\n[{iteration}/{len(origins)}] Sensor origin {origin_id}: {origin}", flush=True)
                load_start = perf_counter()
                fill_patchwork_input(args.input_las, points, origin, args.chunk_size)
                print(f"  LAS reload/translation: {perf_counter() - load_start:.2f}s", flush=True)

                params = pwp.Parameters()
                params.min_range = args.min_range
                params.num_sectors_each_zone = [args.zone0_sectors, 32, 54, 32]
                params.enable_RNR = False
                params.verbose = False
                algorithm = pwp.patchworkpp(params)
                try:
                    t0 = perf_counter()
                    algorithm.estimateGround(points)
                    seconds = perf_counter() - t0
                    ground = np.asarray(algorithm.getGroundIndices(), dtype=np.int64).ravel()
                    nonground_n = np.asarray(algorithm.getNongroundIndices()).size

                    if ground.size and (ground.min() < 0 or ground.max() >= n):
                        raise RuntimeError("Invalid point indices returned by Patchwork++")
                    # A ground index may already have been found by another origin.
                    newly_found = int(np.count_nonzero(~union[ground]))
                    union[ground] = True
                    cumulative = int(np.count_nonzero(union))
                    ground_n = int(ground.size)
                    unclassified_n = n - ground_n - int(nonground_n)
                    if unclassified_n < 0:
                        raise RuntimeError("Inconsistent classification index counts")

                    if iteration == 1 and not args.skip_las_output:
                        first_origin_mask = union.copy()

                    metric = {
                        "origin_id": origin_id,
                        "origin_x": float(origin[0]), "origin_y": float(origin[1]), "origin_z": float(origin[2]),
                        "ground_points": ground_n,
                        "nonground_points": int(nonground_n),
                        "unclassified_points": int(unclassified_n),
                        "new_ground_points": newly_found,
                        "cumulative_ground_points": cumulative,
                        "cumulative_ground_pct": round(cumulative / n * 100, 3),
                        "segmentation_seconds": round(seconds, 3),
                    }
                    writer.writerow(metric)
                    f.flush()
                    print(
                        f"  Ground={ground_n:,}; non-ground={nonground_n:,}; "
                        f"unclassified={unclassified_n:,}; new={newly_found:,}; "
                        f"union={cumulative:,} ({cumulative / n:.2%}); "
                        f"segmentation={seconds:.2f}s",
                        flush=True,
                    )
                finally:
                    del algorithm
                    if "ground" in locals():
                        del ground
                    gc.collect()
    finally:
        del points
        gc.collect()
        try:
            scratch.unlink(missing_ok=True)
        except PermissionError:
            print(f"Warning: delete temporary file manually: {scratch}", flush=True)

    if not args.skip_las_output:
        if first_origin_mask is not None:
            save_ground_subset(args.input_las, args.output_dir / "origin_01_ground.las", first_origin_mask, args.chunk_size)
        save_ground_subset(args.input_las, args.output_dir / "ground_union.las", union, args.chunk_size)
    print(f"\nDone. Total time: {perf_counter() - total_start:.1f}s", flush=True)
    print(f"Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
