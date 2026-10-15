# Checkpoint inference

This guide describes the SkelMo checkpoint, its matching input
contract, and the supported inference entry point.

## Bundle and integrity

Download the [checkpoint bundle](https://drive.google.com/file/d/1ZJ4BKAkn4ms21tmoiNAxJtbXDT3GmKtZ/view?usp=sharing)
and extract it under `checkpoints/`:

```bash
mkdir -p checkpoints
tar -xzf skelmo_checkpoint_bundle.tar.gz -C checkpoints
```

The archive contains these four files and `SHA256SUMS` (the model is 284 MB;
the others are small). `training_stats.json` and `pca_128.npz` are actual files,
not symlinks into a private training dataset.

```text
checkpoints/skelmo/
├── model000099999.pt
├── args.json
├── training_stats.json
└── pca_128.npz
```

The audited source files have these SHA-256 checksums:

```text
d662b18663c1d6ae5ba547adbec927e0d9854291b3e4615b146b1cbc628a4547  model000099999.pt
e6e5139536e57d08b76a56c479fc97608585de1fc7fd5f02b911c71777fa7ee6  args.json
c62f631c9cd66b210a955219031a0e00155d9eee9e12f6fa2fc69dd7c289c640  training_stats.json
ea6b3e778f1a08333525e4edefba882b4bf1c9f41f7ed4fed61687ab2c2bae6a  pca_128.npz
```

The archive SHA-256 is
`9b2bb83ca73aa606447601f192697f2e8a2cdfd42821e9d8bf2c33181c0aa8ec`.

The model uses 32 frames, 100 diffusion steps, pure conditional sampling,
`first_frame_delta` XYZ targets, full spatial/temporal/reverse video attention,
and fixed residual scale 0.1. It has no classifier-free dropout, so use
`--guidance-scale 1`; the new inference entry point rejects other values.
The entry point also verifies all four bundle checksums before sampling so a
different normalization or PCA basis cannot be silently mixed with the model.

Publish the **matching source code** with the weights. The inference path
depends on the current dense-video attention, first-frame conditioning, and
diffusion implementation in `model/`, `data_loaders/`, `diffusion/`, and
`utils/model_util.py`; the older public code revision is not a drop-in loader
for this checkpoint. The release source changes must be committed and tested
before announcing a public download. The four-file bundle alone is not a
complete release.

## Prepared input

Save one `.npz` with these required arrays, all in **the same joint order**:

| Key | Shape | Meaning |
| --- | --- | --- |
| `parents` | `[J]` int | Parent-first hierarchy; one root at joint 0; `parents[0]=-1`; 2–143 joints. |
| `first_pose` | `[J,3]` float | Observed frame-zero joint positions in scene-normalized gravity-camera coordinates. |
| `joint_uv` | `[J,2]` float | Projection of that **same** frame-zero pose into the **same uncropped render** used for DINO. |
| `joint_visible` | `[J]` bool | Joint inside image bounds. This is in-frame visibility, not an occlusion label. |
| `resolution` | scalar int | Width and height of each square input image in pixels. |

Optional: `joint_names` (`[J]` strings) and `fps` (scalar). If omitted, names
default to `joint_0` etc. and IK uses first-frame bone offsets. `dense_dino`
(`[T,1369,128]`) may be included in the
NPZ, supplied separately with `--features`, or generated from RGB frames with
`--frames-dir`. `T` is 1–32. The RGB filenames must sort in temporal order
(e.g. `0000.png`, `0001.png`, …); all frames must have the declared square
resolution. DINO processes the **whole** image resized to 518×518 and the
supplied 128-D PCA basis. Do not crop the person without also updating camera
projection and `joint_uv`; keeping the full render is the verified protocol.
The optional rest-pose appearance checkpoint also requires `rest_pose`
(`[J,3]`) and `rest_appearance` (`[J,768]`) in the NPZ, as described in the
folded README section.

The coordinate system is X-left, Y-up, Z-forward in the training
gravity-camera frame. The longest source scene bounding-box dimension is
normalized to one unit; the output is **not automatically in physical metres**.
The first pose, camera extrinsics/intrinsics, UV, and rendered RGB must describe
the same scene and first frame. The script cannot estimate camera calibration
or the frame-zero 3D skeleton from an arbitrary video.

## Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m sample.generate_checkpoint \
  --checkpoint checkpoints/skelmo/model000099999.pt \
  --stats checkpoints/skelmo/training_stats.json \
  --pca checkpoints/skelmo/pca_128.npz \
  --input /path/to/firstframe_input.npz \
  --frames-dir /path/to/uncropped_square_frames \
  --seed 2026 --guidance-scale 1 \
  --ik-bvh --output outputs/example
```

`--features /path/to/dense_dino.npy` replaces `--frames-dir` for a precomputed
PCA128 cache. The command writes `prediction.npz` (raw XYZ plus hierarchy),
`summary.json`, and optionally `prediction_ik.bvh` and `prediction_ik.npy`.
With RGB frames it also writes `first_frame_projection.png`: inspect it before
trusting the motion. Bones should overlay the person in the first image.
The output directory must not already exist, preventing accidental overwrite.
The BVH is still in model camera coordinates; to apply it to the input GLB,
undo the scene/camera normalization and retarget by the supplied joint names.

## Preparing a new skeleton

1. Obtain the evaluated/deformed bone-head positions for the **first video
   frame** from the rig, along with the parent hierarchy and joint names.
   Remove helper/end joints consistently from every array, and retain one
   parent-first root. A rest BVH alone is insufficient: this checkpoint is
   explicitly conditioned on the observed first pose.
2. Render 1–32 frames with a fixed, known camera and use the same scene
   normalization for both rendered mesh and skeleton. The project's dataset
   implementation is in `preprocess/static_camera_pilot/build_dataset.py`:
   it stores the scene transform, camera intrinsics/extrinsics, evaluated
   gravity-camera joint positions, 2D projections, and in-frame mask.
3. Project the frame-zero joints through that exact render camera. Save
   `first_pose`, `joint_uv`, `joint_visible`, `parents`, and `resolution` in
   `firstframe_input.npz`. Do not use the source BVH's unnormalized coordinates
   as `first_pose` or project with a guessed FOV/distance.
4. Supply the matching uncropped RGB frames to the command above. Check
   `first_frame_projection.png`. If the overlay is off, fix the camera,
   coordinate-axis conversion, scene scale, joint order, or crop **before**
   interpreting model quality. The model never receives future 3D poses.

## Driving video: required preparation

This checkpoint does **not** take an arbitrary MP4 by itself. The verified
route starts with a rigged character, its observed frame-zero 3D pose, and a
fixed, calibrated render camera. Frame zero of the driving sequence must show
that same posed character from that same camera. A monocular video without
known camera and frame-zero 3D pose needs a separate estimation/calibration
stage; this release does not claim to solve it automatically.

1. Render the posed character and its subsequent motion with one fixed camera.
   Save the exact render intrinsics, world-to-camera transform, scene
   normalization, and original frame rate. Keep the *entire* square render in
   every frame. If the source is rectangular, prepare a calibrated square
   render/letterbox and apply its exact pixel transform to the joint
   projections; do not independently crop or stretch the frames.
2. Export 1–32 consecutive RGB frames in temporal order. A zero-padded frame
   sequence such as `0000.png`, `0001.png`, … is accepted by `--frames-dir`.
   To extract the first 32 frames of an already calibrated square video:

   ```bash
   mkdir -p /path/to/frames
   ffmpeg -i /path/to/driving.mp4 -map 0:v:0 -frames:v 32 \
     -start_number 0 /path/to/frames/%04d.png
   ```

   Check that extraction preserved frame order and the intended timestamps;
   save the source frame rate in the optional `fps` NPZ field. Do not insert a
   separate rest-pose image or duplicate frame zero. The model's reference
   token is constructed internally from frame zero.
3. Evaluate the rig's joint positions at the *same* first frame. Transform
   them into the scene-normalized gravity-camera frame used by the model, then
   project them with the *exact* render camera into pixel coordinates of the
   delivered frame. Store these as `first_pose` and `joint_uv`, together with
   parent-first `parents`, an in-frame `joint_visible` mask, and square
   `resolution`. The arrays must have identical joint order. `joint_visible`
   is not an occlusion test; hidden-but-in-frame joints retain their projected
   coordinates. See the prepared-input table above.
4. Run inference with `--frames-dir`. The CLI applies the verified DINOv2
   ViT-B/14 preprocessing to each **whole** RGB frame (resize to 518×518,
   ImageNet normalization, 37×37 patch tokens), then projects the 768-D patch
   descriptors to PCA128 using the released `pca_128.npz`. Do not feed a
   different feature cache or append a CLS token. Alternatively, provide a
   precomputed `[T,1369,128]` cache from exactly this pipeline using
   `--features`.
5. Inspect `first_frame_projection.png` before using the result. Every bone
   segment and visible joint should overlay the person in `0000.png`. A
   flipped walking direction, shifted skeleton, or mismatched body scale is
   a reason to recheck camera axes, FOV, render distance, scene scale, and
   pixel transforms before judging model quality. The script validates array
   shapes and bounds, but only visual inspection can confirm the semantic
   overlay.

The generated BVH is in the model's normalized camera frame. To animate the
original GLB, undo the recorded scene/camera transform and retarget by joint
name/order; IK only enforces the supplied first-frame bone lengths. Longer
videos require a separate windowing and temporal stitching procedure, which
is not part of this release.

Only one 32-frame window is supported by this release entry point. Stitching
longer videos requires an explicit temporal consistency policy and is not
claimed here.
