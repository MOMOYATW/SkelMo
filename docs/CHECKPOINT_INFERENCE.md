# Checkpoint inference

## Download

Download the [checkpoint bundle](https://drive.google.com/file/d/1ZJ4BKAkn4ms21tmoiNAxJtbXDT3GmKtZ/view?usp=sharing) and extract it from the repository root:

```bash
mkdir -p checkpoints
tar -xzf skelmo_checkpoint_bundle.tar.gz -C checkpoints
```

The bundle contains `model000099999.pt`, `args.json`, `training_stats.json`, `pca_128.npz`, and `SHA256SUMS` under `checkpoints/skelmo/`. The inference command verifies the matching model, normalization, and PCA files. Use `--guidance-scale 1`: this checkpoint was trained without unconditional dropout.

## Animated rigged GLB

Use Python with Blender 4.x `bpy` for the first command, then the `skelmo` environment from `environment.yaml` for the second:

```bash
python -m preprocess.static_camera_pilot.render_checkpoint_input \
  --glb /path/to/animated_character.glb \
  --output inputs/character_run \
  --frames 32 --fps 30 --resolution 512 --azimuth -90

CUDA_VISIBLE_DEVICES=0 python -m sample.generate_checkpoint \
  --checkpoint checkpoints/skelmo/model000099999.pt \
  --stats checkpoints/skelmo/training_stats.json \
  --pca checkpoints/skelmo/pca_128.npz \
  --input inputs/character_run/firstframe_input.npz \
  --frames-dir inputs/character_run/frames \
  --seed 2026 --guidance-scale 1 --output outputs/character_run
```

The renderer saves uncropped RGB frames, `camera.json`, `firstframe_input.npz`, and `first_frame_projection.png`. Check that the projected skeleton overlays the person in frame zero before sampling. The sampler writes `prediction.npz` and `summary.json`; add `--ik-bvh` for a bone-length-constrained BVH. Neither command overwrites an existing output directory.

For a GLB with multiple actions, select one with `--action NAME`. Use `--start-frame` for another window or `--joints-json` for a parent-first JSON list of bones when helper/end bones must be excluded. Keep `camera.json` for inverse transforms and retargeting.

## Separately prepared driving video

An arbitrary MP4 is **not** sufficient input. Supply a fixed calibrated camera and the observed 3D skeleton at the video's first frame. The frames and skeleton must use the same scene transform, camera, and pixel coordinates. The released tools do not recover camera calibration or a first-frame 3D pose from monocular video.

Extract 1–32 ordered, square, uncropped frames (for example `0000.png`, `0001.png`, …). Prepare a `.npz` with arrays in one parent-first joint order:

| Key | Shape | Content |
| --- | --- | --- |
| `parents` | `[J]` | One root at joint 0 (`-1`); 2–143 joints. |
| `first_pose` | `[J,3]` | Frame-zero joints in normalized gravity-camera coordinates. |
| `joint_uv` | `[J,2]` | Those joints projected into frame zero, in pixels. |
| `joint_visible` | `[J]` | In-frame mask, not an occlusion label. |
| `resolution` | scalar | Width and height of each square frame. |

Optional `joint_names [J]` and `fps` are supported. The model coordinates are X-left, Y-up, Z-forward; the longest source-scene bounding-box dimension is normalized to one, so output units are not automatically metres. If you crop, resize non-uniformly, or change FOV, update `joint_uv` and the camera transform accordingly. The supplied first pose must be the observed pose, not a rest pose.

Run the same sampling command above with `--input /path/to/firstframe_input.npz` and `--frames-dir /path/to/frames`. It extracts DINOv2 ViT-B/14 features from each **whole** frame using the bundle's PCA basis. Alternatively, provide matching `[T,1369,128]` features with `--features /path/to/dense_dino.npy`. Inspect the generated `first_frame_projection.png`; misalignment usually means a camera, scale, axis, crop, or joint-order mismatch.

Inference covers one window of at most 32 frames. Longer videos need a separate windowing and stitching policy. The optional BVH remains in normalized model-camera coordinates; applying it to the source asset requires inverse scene/camera transforms and retargeting by joint name.

The optional rest-pose appearance checkpoint and its extra inputs are described in the folded section of the [README](../README.md).
