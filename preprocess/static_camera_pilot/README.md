# Static-camera pilot

This pilot deliberately uses one static camera per clip. Its yaw is sampled
deterministically and uniformly over 360 degrees from the clip id and camera
seed, while elevation remains zero. Moving cameras are deferred until rendered
backgrounds provide enough ego-motion evidence.

The source is the local Objaverse-XL Rigged Animated collection. Selection is
one motion per asset, round-robin balanced by raw joint-count bins, so 1,024
clips also means 1,024 distinct rigs.

```bash
python -m preprocess.static_camera_pilot.select_manifest \
  --dataset-root /home/ytao/Experiment/UniMate/dataset/Objaverse-XL-Rigged-Animated \
  --output /home/ytao/Datasets/skelmo_static_pilot_v2_random_yaw/selection.jsonl
```

The initial render contract is 512x512, 30 fps, at most 128 frames, transparent
WebP frames plus an H.264 inspection video. Every clip also has a separate
`rest.webp`, rendered after the action with the exact same fixed camera and
action-derived scene normalization. REST is deliberately not inserted into the
MP4 or frame directory, so video and motion label lengths remain identical.
Every clip stores the fixed camera intrinsics/extrinsics and both normalized
world and gravity-camera targets.

Build a smoke set with UniMate's Blender-enabled Python:

```bash
CUDA_VISIBLE_DEVICES=7 /home/ytao/miniconda3/envs/unimate/bin/python \
  -m preprocess.static_camera_pilot.build_dataset \
  --selection /home/ytao/Datasets/skelmo_static_pilot_v2_random_yaw/selection.jsonl \
  --dataset-root /home/ytao/Experiment/UniMate/dataset/Objaverse-XL-Rigged-Animated \
  --unimate-root /home/ytao/Experiment/UniMate \
  --output /home/ytao/Datasets/skelmo_static_pilot_v2_random_yaw/smoke \
  --limit 1 --resolution 256 --max-frames 32 --samples 16
```

The primary rotation target is the rest-relative global skinning deformation
expressed in the gravity-aligned camera frame. Both its full 3x3 matrix and an
explicit column-layout 6D representation are retained during the pilot. This
avoids silently inheriting the row/column ambiguity of older 6D utilities and
lets us audit twist before choosing the final compact representation.

The skeletal model sequence is `[REST_GEOMETRY, F0, F1, ...]`. REST_GEOMETRY
comes from the numeric rig structure in the same gravity-camera coordinate
system as the motion. The visual stream is `[NULL, video_F0, video_F1, ...]`:
no rendered REST image is encoded or passed to the model, and frame zero is
not duplicated. Existing `rest.webp` files are inspection artifacts only.
Absolute camera-space root placement is preserved.

Each joint also receives an offline appearance descriptor that is independent
of the motion video. `extract_rest_joint_features` renders the animation-free
mesh from six axis-aligned and eight diagonal views, samples DINOv2 features
back onto visible vertices, and pools them to joints with skinning weights.
At load time the 768D appearance descriptor is projected through the same PCA
basis as the video patches and concatenated with a 128D numeric topology
descriptor. The 14 rest renders are preprocessing artifacts, never video
frames.

The deployable visual condition is a cached dense DINO grid rather than a
ground-truth joint track. DINOv2-B/14 produces 37x37x768 tokens per frame; a
shared 128D PCA basis preserves the spatial grid while reducing this pilot to
about 31 GiB. Fit the basis once, then extract restartable shards:

```bash
python -m preprocess.static_camera_pilot.extract_dense_dino fit-pca \
  --root /path/to/full --components 128
python -m preprocess.static_camera_pilot.extract_dense_dino extract \
  --root /path/to/full --components 128 --num-shards 4 --shard-index 0
```

The model uses rest appearance+topology as joint queries over these dense
patches. No ground-truth 2D joint positions are required at inference time.

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
  --root /home/ytao/Datasets/skelmo_static_pilot_v2_random_yaw/smoke --expected 8
```

For direct pixel/label inspection, `render_overlay` draws the projected bones
over any chosen WebP frame.

Generate and serve an inspection gallery containing REST, motion video, and
projected-label overlays:

```bash
python -m preprocess.static_camera_pilot.make_gallery --root /path/to/smoke
python -m preprocess.static_camera_pilot.serve_gallery \
  --root /path/to/smoke --port 8766
```
