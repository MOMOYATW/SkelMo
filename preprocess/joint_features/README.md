# Rest-pose joint features

This pipeline is only for the optional rest-pose appearance checkpoint. The main checkpoint does not need it.

It requires a rigged GLB with mesh, armature, and skinning weights; Blender 4.x; and the Python environment in `environment.yaml`. From the repository root:

```bash
python -m preprocess.joint_features.pipeline \
  --input-dir /path/to/rigged_glbs \
  --render-root /path/to/work/renders \
  --output-root /path/to/work/features
```

The pipeline renders six axis-aligned and eight diagonal views of the animation-free mesh, extracts DINOv2 ViT-B/14 descriptors, and pools them to joints using skinning weights. It normally groups filenames by the prefix before the first underscore and prefers a rest/T-pose/idle file; use `--input-list` to choose files explicitly.

For each asset, the output includes `joint_max_weight_features.npy` and `joint_averaged_features.npy`, both keyed by joint name. Match names to the prepared skeleton's joint order, use the max-weight descriptor when present and the averaged descriptor as fallback, and stack them into `rest_appearance [J,768]`. Also supply `rest_pose [J,3]` in the same normalized camera frame and scale as `first_pose`; see the optional checkpoint section of the [README](../../README.md).

Use the default DINOv2 encoder for this checkpoint; another encoder is not interchangeable. Load the generated `.npy` dictionaries only from a trusted source because they use NumPy pickle objects.
