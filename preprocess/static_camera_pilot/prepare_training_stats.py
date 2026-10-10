"""Compute train-split normalization statistics for static-camera positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    total = np.zeros(3, dtype=np.float64)
    square = np.zeros(3, dtype=np.float64)
    count = 0
    records = 0
    for path in sorted((args.root / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("split") != "train":
            continue
        with np.load(record["motion_path"], allow_pickle=False) as motion:
            positions = motion["joint_position_gravity_camera"].astype(np.float64)
        flat = positions.reshape(-1, 3)
        total += flat.sum(0)
        square += np.square(flat).sum(0)
        count += len(flat)
        records += 1
    if count == 0:
        raise ValueError("No train positions found")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 0.0))
    payload = {
        "format": "skelmo_static_training_stats_v1",
        "records": records,
        "samples": count,
        "position_mean": mean.tolist(),
        "position_std": std.tolist(),
        "rotation_mean": [0.0] * 6,
        "rotation_std": [1.0] * 6,
    }
    output = args.root / "training_stats.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
