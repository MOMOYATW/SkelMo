"""Build a self-contained index page for visual QA of a pilot dataset."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote


def _relative_url(path, root):
    return quote(str(Path(path).resolve().relative_to(root.resolve())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "index.html"
    records = [json.loads(path.read_text()) for path in sorted((root / "records").glob("*.json"))]
    cards = []
    for record in records:
        clip_id = record["clip_id"]
        render_dir = Path(record["render_dir"])
        rest_path = Path(record.get("rest_render_path", render_dir / "rest.webp"))
        video_path = render_dir / "v000.mp4"
        rest_overlay = root / "overlays" / f"{clip_id}_rest.webp"
        motion_overlay = root / "overlays" / f"{clip_id}_f000.webp"
        metadata = (
            f"{record['joint_count']} joints · {record['frame_count']} frames · "
            f"yaw {float(record.get('camera_azimuth_deg', 0.0)):+.1f}° · "
            f"{record.get('split', 'unknown')}"
        )
        overlay_html = ""
        if rest_overlay.is_file() and motion_overlay.is_file():
            overlay_html = f"""
              <details><summary>Projected-label audit</summary>
                <div class="pair">
                  <figure><img src="{_relative_url(rest_overlay, root)}"><figcaption>REST + projected skeleton</figcaption></figure>
                  <figure><img src="{_relative_url(motion_overlay, root)}"><figcaption>F0 + projected skeleton</figcaption></figure>
                </div>
              </details>"""
        cards.append(f"""
          <article>
            <h2>{html.escape(clip_id)}</h2>
            <p>{html.escape(metadata)}</p>
            <div class="pair">
              <figure><img src="{_relative_url(rest_path, root)}"><figcaption>True REST conditioning image</figcaption></figure>
              <figure><video src="{_relative_url(video_path, root)}" controls loop muted autoplay playsinline></video><figcaption>Motion F0…FT (REST not inserted)</figcaption></figure>
            </div>
            {overlay_html}
          </article>""")
    validation = {}
    validation_path = root / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text())
    status = (
        f"{validation.get('records', len(records))} clips · "
        f"{validation.get('unique_rigs', '?')} unique rigs · "
        f"{len(validation.get('failures', {}))} validation failures"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkelMo random-yaw static-camera pilot</title>
<style>
body{{margin:0;background:#11151a;color:#e9eef4;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}.summary{{color:#aab8c5;margin:0 0 24px}}article{{background:#1a2028;border:1px solid #303a45;border-radius:14px;padding:18px;margin:18px 0}}h2{{font-size:16px;margin:0}}article>p{{color:#93a4b5;margin:4px 0 14px}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0}}img,video{{display:block;width:100%;aspect-ratio:1;object-fit:contain;border-radius:9px;background-color:#dae0e5;background-image:linear-gradient(45deg,#c8d0d8 25%,transparent 25%),linear-gradient(-45deg,#c8d0d8 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#c8d0d8 75%),linear-gradient(-45deg,transparent 75%,#c8d0d8 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}}figcaption{{color:#b8c5d1;margin-top:7px}}details{{margin-top:14px}}summary{{cursor:pointer;color:#8fc6ff}}@media(max-width:720px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>SkelMo random-yaw static-camera pilot</h1><p class="summary">{html.escape(status)}. Each clip samples one deterministic yaw; REST and motion share the exact camera, normalization, crop, and transparent background.</p>{''.join(cards)}</main></body></html>"""
    output.write_text(document)
    print(output)


if __name__ == "__main__":
    main()
