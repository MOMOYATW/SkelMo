# Rest-pose joint features

This pipeline is only for the optional rest-pose appearance checkpoint. The main checkpoint instead samples first-frame PCA-DINO descriptors at the projected joint pixels; it uses neither of the `.npy` files below for training or inference.

It requires a rigged GLB with mesh, armature, and skinning weights; Blender 4.x; and the Python environment in `environment.yaml`. From the repository root:

```bash
python -m preprocess.joint_features.pipeline \
  --input-dir /path/to/rigged_glbs \
  --render-root /path/to/work/renders \
  --output-root /path/to/work/features
```

The pipeline renders six axis-aligned and eight diagonal views of the animation-free mesh, extracts DINOv2 ViT-B/14 descriptors, and pools them to joints using skinning weights. It normally groups filenames by the prefix before the first underscore and prefers a rest/T-pose/idle file; use `--input-list` to choose files explicitly.

For each asset, the output includes `joint_max_weight_features.npy` and `joint_averaged_features.npy`, both keyed by joint name. The optional rest-pose checkpoint was trained with the max-weight descriptor where available and the averaged descriptor only as fallback. To use that checkpoint, reproduce this choice in the prepared skeleton's joint order and stack the descriptors into `rest_appearance [J,768]`. Also supply `rest_pose [J,3]` in the same normalized camera frame and scale as `first_pose`; see the optional checkpoint section of the [README](../../README.md). Neither descriptor file is an input to the main checkpoint.

Use the default DINOv2 encoder for this checkpoint; another encoder is not interchangeable. Load the generated `.npy` dictionaries only from a trusted source because they use NumPy pickle objects.
