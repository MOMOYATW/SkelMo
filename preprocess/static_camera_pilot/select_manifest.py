"""Select a deterministic, rig-diverse static-camera pilot from Objaverse GLBs."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


JOINT_BINS = (0, 16, 24, 32, 48, 64, 96, 144)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def joint_bin(count):
    for low, high in zip(JOINT_BINS[:-1], JOINT_BINS[1:]):
        if low < count <= high:
            return f"{low + 1:03d}-{high:03d}"
    return "overflow"


def stable_value(seed, *parts):
    payload = "|".join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def select_records(dataset_root, count, seed, min_frames, max_joints):
    metadata = {
        row["object_id"]: row
        for row in read_csv(dataset_root / "metadata.csv")
    }
    excluded = {
        row["object_id"]
        for row in read_csv(dataset_root / "excluded.csv")
        if row.get("object_id")
    }
    by_asset = defaultdict(list)
    for row in read_csv(dataset_root / "animations.csv"):
        object_id = row["object_id"]
        meta = metadata.get(object_id)
        frames = int(round(float(row["duration_sec"]) * 30.0)) + 1
        joints = int(meta["num_joints"]) if meta else 0
        if (
            object_id in excluded
            or meta is None
            or row["drives_skeleton"].lower() != "true"
            or frames < min_frames
            or not 2 <= joints <= max_joints
            or not (dataset_root / row["file"]).is_file()
        ):
            continue
        candidate = dict(row)
        candidate["source_frames_30fps"] = frames
        candidate["num_joints_raw"] = joints
        candidate["num_vertices"] = int(meta["num_vertices"])
        candidate["num_meshes"] = int(meta["num_meshes"])
        candidate["joint_bin"] = joint_bin(joints)
        by_asset[object_id].append(candidate)

    # One clip per asset maximizes rig/topology coverage. Prefer a substantial
    # motion over a tiny idle, with a seeded hash resolving comparable clips.
    candidates = []
    for object_id, clips in by_asset.items():
        clips.sort(
            key=lambda row: (
                min(row["source_frames_30fps"], 128),
                int(row["animated_joints"]),
                stable_value(seed, object_id, row["clip_index"]),
            ),
            reverse=True,
        )
        candidates.append(clips[0])

    buckets = defaultdict(list)
    for row in candidates:
        buckets[row["joint_bin"]].append(row)
    for name, rows in buckets.items():
        rows.sort(key=lambda row: stable_value(seed, name, row["object_id"]))

    # Round-robin across joint-count bins instead of allowing the humanoid
    # 24-64-joint majority to consume the entire pilot.
    selected = []
    names = sorted(buckets)
    cursor = 0
    while len(selected) < count and any(buckets.values()):
        name = names[cursor % len(names)]
        cursor += 1
        if buckets[name]:
            selected.append(buckets[name].pop())
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} eligible rig-diverse clips; requested {count}")

    rng = random.Random(seed)
    rng.shuffle(selected)
    split_counts = {
        "train": int(round(count * 0.8)),
        "val": int(round(count * 0.1)),
    }
    split_counts["test"] = count - split_counts["train"] - split_counts["val"]
    boundaries = (
        split_counts["train"],
        split_counts["train"] + split_counts["val"],
    )
    for index, row in enumerate(selected):
        split = "train" if index < boundaries[0] else "val" if index < boundaries[1] else "test"
        row.update(
            {
                "format": "skelmo_static_camera_selection_v1",
                "clip_id": f"{row['object_id']}_a{int(row['clip_index']):03d}",
                "rig_id": row["object_id"],
                "split": split,
                "fps": 30,
                "max_frames": 128,
                "camera": "static_front",
            }
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=1024, type=int)
    parser.add_argument("--seed", default=20261010, type=int)
    parser.add_argument("--min-frames", default=32, type=int)
    parser.add_argument("--max-joints", default=143, type=int)
    args = parser.parse_args()
    rows = select_records(
        args.dataset_root.resolve(), args.count, args.seed,
        args.min_frames, args.max_joints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "format": "skelmo_static_camera_selection_summary_v1",
        "count": len(rows),
        "seed": args.seed,
        "camera": "static_front",
        "fps": 30,
        "max_frames": 128,
        "splits": Counter(row["split"] for row in rows),
        "joint_bins": Counter(row["joint_bin"] for row in rows),
        "unique_rigs": len({row["rig_id"] for row in rows}),
        "total_capped_frames": sum(min(row["source_frames_30fps"], 128) for row in rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
