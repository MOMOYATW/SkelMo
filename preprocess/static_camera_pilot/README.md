# Static-camera pilot

This pilot deliberately uses one fixed front camera per clip. Moving cameras
are deferred until rendered backgrounds provide enough ego-motion evidence.

The source is the local Objaverse-XL Rigged Animated collection. Selection is
one motion per asset, round-robin balanced by raw joint-count bins, so 1,024
clips also means 1,024 distinct rigs.

```bash
python -m preprocess.static_camera_pilot.select_manifest \
  --dataset-root /home/ytao/Experiment/UniMate/dataset/Objaverse-XL-Rigged-Animated \
  --output /home/ytao/Datasets/skelmo_static_pilot_v1/selection.jsonl
```

The initial render contract is 512x512, 30 fps, at most 128 frames, transparent
WebP frames plus an H.264 inspection video. Every clip stores the fixed camera
intrinsics/extrinsics and both normalized world and gravity-camera targets.

Build a smoke set with UniMate's Blender-enabled Python:

```bash
CUDA_VISIBLE_DEVICES=7 /home/ytao/miniconda3/envs/unimate/bin/python \
  -m preprocess.static_camera_pilot.build_dataset \
  --selection /home/ytao/Datasets/skelmo_static_pilot_v1/selection.jsonl \
  --dataset-root /home/ytao/Experiment/UniMate/dataset/Objaverse-XL-Rigged-Animated \
  --unimate-root /home/ytao/Experiment/UniMate \
  --output /home/ytao/Datasets/skelmo_static_pilot_v1/smoke \
  --limit 1 --resolution 256 --max-frames 32 --samples 16
```

The primary rotation target is the rest-relative global skinning deformation
expressed in the gravity-aligned camera frame. Both its full 3x3 matrix and an
explicit column-layout 6D representation are retained during the pilot. This
avoids silently inheriting the row/column ambiguity of older 6D utilities and
lets us audit twist before choosing the final compact representation.

`rigs/*.npz` stays in unnormalized Y-up source coordinates. Per-clip render
normalization and the fixed camera live in `motions/*.npz`; therefore adding a
second motion for the same rig cannot overwrite its canonical rig geometry.

Pixel-aligned joint positions and global rotations are read from Blender's
evaluated pose matrices, including the armature object's scale. The portable
UniMate export-space local transforms are retained under explicit `*_export_*`
keys, but are not used as video-space position targets because many GLBs carry
non-unit object scale.

Validate every completed record before training:

```bash
/home/ytao/miniconda3/envs/anytop/bin/python \
  -m preprocess.static_camera_pilot.validate_dataset \
  --root /home/ytao/Datasets/skelmo_static_pilot_v1/smoke --expected 8
```

For direct pixel/label inspection, `render_overlay` draws the projected bones
over any chosen WebP frame.
