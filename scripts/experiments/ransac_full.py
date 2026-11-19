import argparse
import gc
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-inliers", type=Path, default=None, help="Save RANSAC inliers as a LAS file")
    args = parser.parse_args()

    if args.save_inliers is not None:
        if args.save_inliers.exists():
            parser.error(f"Output already exists: {args.save_inliers}")
        if args.save_inliers.resolve() == args.input.resolve():
            parser.error("Output path must differ from input path")

    start = perf_counter()

    # Read all coordinates without loading every LAS attribute.
    with laspy.open(args.input) as reader:
        n = reader.header.point_count
        print(f"Points: {n:,}", flush=True)
        print(f"XYZ array: {n * 24 / 2**30:.2f} GiB")

        xyz = np.empty((n, 3), dtype=np.float64)
        offset = 0

        for chunk in reader.chunk_iterator(1_000_000):
            count = len(chunk)
            xyz[offset:offset + count, 0] = chunk.x
            xyz[offset:offset + count, 1] = chunk.y
            xyz[offset:offset + count, 2] = chunk.z
            offset += count

    print(f"Loading: {perf_counter() - start:.2f}s",
          flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    del xyz
    gc.collect()

    print("Running RANSAC...", flush=True)

    o3d.utility.random.seed(42)
    t0 = perf_counter()

    model, inliers = pcd.segment_plane(
        distance_threshold=args.threshold,
        ransac_n=3,
        num_iterations=args.iterations,
        probability=1.0,
    )

    elapsed = perf_counter() - t0

    a, b, c, d = model
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)

    tilt = np.degrees(
        np.arccos(np.clip(abs(c) / norm, 0, 1))
    )

    print("\n=== RANSAC RESULT ===")
    print(f"Plane: {model}")
    print(f"Tilt from horizontal: {tilt:.3f} degrees")
    print(f"Inliers: {len(inliers):,}")
    print(f"Inlier ratio: {len(inliers) / n:.2%}")
    print(f"RANSAC time: {elapsed:.2f}s")
    print(f"Total time: {perf_counter() - start:.2f}s")

    if args.save_inliers is not None:
        save_inliers_las(
            args.input,
            args.save_inliers,
            inliers,
            n,
        )


def save_inliers_las(input_path, output_path, inliers, total_points):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map RANSAC indices to the original LAS point order.
    selected = np.zeros(total_points, dtype=bool)
    selected[np.asarray(inliers, dtype=np.int64)] = True

    offset = 0

    with laspy.open(input_path) as reader:
        with laspy.open(output_path, mode="w", header=reader.header) as writer:
            for chunk in reader.chunk_iterator(1_000_000):
                count = len(chunk)
                mask = selected[offset:offset + count]

                if np.any(mask):
                    writer.write_points(chunk[mask])

                offset += count

    print(f"Saved {len(inliers):,} inliers to: {output_path}")


if __name__ == "__main__":
    main()
