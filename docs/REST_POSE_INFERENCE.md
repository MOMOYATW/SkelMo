# Optional rest-pose checkpoint inference

This checkpoint replaces the per-joint first-frame DINO descriptors with skin-weighted **averages** from 14 rest-pose views. It still requires the aligned first-frame 3D skeleton and uncropped driving frames described in the [main inference guide](CHECKPOINT_INFERENCE.md). The max-weight descriptors are not used.

Download the [rest-pose checkpoint bundle](https://drive.google.com/file/d/1Lo4tB7-T5Lsm9JwOwbRDMF4jhMSTsOXD/view?usp=sharing) and extract it at the repository root:

```bash
mkdir -p checkpoints
tar -xzf skelmo_rest_average_checkpoint_bundle.tar.gz -C checkpoints
```

For an animated rigged GLB, first prepare its frames, camera, and skeleton using the renderer in the [main guide](CHECKPOINT_INFERENCE.md). Its `firstframe_input.npz` already includes the rest pose and joint names. Extract the averaged appearance descriptors from the same rig:

```bash
python -m preprocess.joint_features.pipeline \
  --input-dir /path/to/rigged_glbs \
  --render-root /path/to/work/rest_renders \
  --output-root /path/to/work/rest_features
```

The pipeline needs Blender 4.x and the environment from `environment.yaml`. Select the intended asset explicitly with `--asset NAME` or `--input-list` if automatic filename grouping would choose a different GLB. Use its `joint_averaged_features.npy`, matched to the prepared skeleton by joint name:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sample.generate_checkpoint \
  --checkpoint checkpoints/skelmo_rest_average/model000099999.pt \
  --stats checkpoints/skelmo_rest_average/training_stats.json \
  --pca checkpoints/skelmo_rest_average/pca_128.npz \
  --input inputs/character_run/firstframe_input.npz \
  --frames-dir inputs/character_run/frames \
  --rest-features /path/to/work/rest_features/ASSET/joint_averaged_features.npy \
  --seed 2026 --guidance-scale 1 --output outputs/character_rest
```

Check the renderer's `first_frame_projection.png` before inference. The rest pose, first-frame pose, joint names/order, normalization, and camera must refer to the same rig. For a separately prepared driving video, follow the [main guide](CHECKPOINT_INFERENCE.md) and additionally provide `rest_pose [J,3]` in the same normalized gravity-camera coordinates plus `joint_names [J]`; alternatively embed the averaged `rest_appearance [J,768]` in the input NPZ instead of passing `--rest-features`. Only load the pipeline's `.npy` feature dictionary from a trusted source because NumPy pickle is used.
