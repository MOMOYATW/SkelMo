"""Synchronize selection-level metadata into completed dataset records."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


SELECTION_KEYS = (
    "split",
    "joint_bin",
    "source_frames_30fps",
    "camera",
    "fps",
    "max_frames",
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()

    selected = {
        row["clip_id"]: row
        for row in (
            json.loads(line)
            for line in args.selection.read_text().splitlines()
            if line.strip()
        )
    }
    updated = 0
    unchanged = 0
    missing = []
    for clip_id, selection in selected.items():
        path = args.root / "records" / f"{clip_id}.json"
        if not path.is_file():
            missing.append(clip_id)
            continue
        record = json.loads(path.read_text())
        changed = False
        for key in SELECTION_KEYS:
            if key in selection and record.get(key) != selection[key]:
                record[key] = selection[key]
                changed = True
        if changed:
            atomic_json(path, record)
            updated += 1
        else:
            unchanged += 1

    print(json.dumps({
        "selected": len(selected),
        "updated": updated,
        "unchanged": unchanged,
        "missing": len(missing),
        "missing_examples": missing[:10],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
