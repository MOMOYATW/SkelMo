"""Draw projected skeleton labels over a rendered frame for visual QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--frame", default=0, type=int, help="Zero-based clip frame")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    motion = np.load(record["motion_path"])
    rig = np.load(record["rig_path"])
    if not 0 <= args.frame < int(record["frame_count"]):
        parser.error("frame is outside the clip")
    source_frame = int(record["frame_start"]) + args.frame
    frame_path = Path(record["render_dir"]) / "v000" / f"{source_frame:04d}.webp"
    image = Image.open(frame_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    uv = motion["joint_position_2d_px"][args.frame]
    valid = motion["joint_in_frame"][args.frame]
    parents = rig["parents"]
    for joint, parent in enumerate(parents):
        if parent >= 0 and valid[joint] and valid[parent]:
            draw.line([tuple(uv[parent]), tuple(uv[joint])], fill=(255, 70, 40, 255), width=2)
    radius = max(2, image.width // 128)
    for joint, point in enumerate(uv):
        if not valid[joint]:
            continue
        x, y = point
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(30, 240, 255, 255),
            outline=(0, 0, 0, 255),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
