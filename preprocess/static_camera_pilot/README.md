# Static-camera preprocessing

For checkpoint inference from a new animated GLB, use the self-contained `render_checkpoint_input.py` command in the [inference guide](../../docs/CHECKPOINT_INFERENCE.md). The bulk dataset builder below is for training-data preparation and uses a separate UniMate checkout; it is not needed for checkpoint inference.

## Training-data pipeline

1. Select rigged animated assets with `select_manifest.py`.
2. Render each clip and export its rig, motion, camera, and projected joints with `build_dataset.py` (run in a Blender-enabled Python environment with UniMate available).
3. Fit a shared PCA128 basis and extract dense video features with `extract_dense_dino.py`.
4. Check completed records with `validate_dataset.py`; inspect projections with `make_gallery.py` when needed.

Example commands from the repository root:

```bash
python -m preprocess.static_camera_pilot.select_manifest \
  --dataset-root /path/to/rigged_assets \
  --output /path/to/selection.jsonl

python -m preprocess.static_camera_pilot.build_dataset \
  --selection /path/to/selection.jsonl \
  --dataset-root /path/to/rigged_assets \
  --unimate-root /path/to/UniMate \
  --output /path/to/dataset

python -m preprocess.static_camera_pilot.extract_dense_dino fit-pca \
  --root /path/to/dataset --components 128
python -m preprocess.static_camera_pilot.extract_dense_dino extract \
  --root /path/to/dataset --components 128
```

Each clip uses one fixed camera. Its rendered frames, frame-zero skeleton, camera metadata, and 2D projections must share the same scene normalization and joint order. A separate rest image may be saved for inspection, but is not inserted into the motion video. The released main checkpoint conditions its joints on the first video frame; 14-view rest-pose appearance is only used by the [optional variant](../joint_features/README.md).
