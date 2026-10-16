# Rest-pose joint feature preprocessing

This preprocessing supplies averaged rest-pose appearance descriptors for the [optional rest-pose checkpoint](../../docs/REST_POSE_INFERENCE.md). The default checkpoint instead samples first-frame PCA-DINO descriptors at projected joint pixels.

It requires a rigged GLB with mesh, armature, and skinning weights; Blender 4.x; and the Python environment in `environment.yaml`. From the repository root:

```bash
python -m preprocess.joint_features.pipeline \
  --input-dir /path/to/rigged_glbs \
  --render-root /path/to/work/renders \
  --output-root /path/to/work/features
```

The pipeline renders six axis-aligned and eight diagonal views of the animation-free mesh, extracts DINOv2 ViT-B/14 descriptors, and pools them to joints using skinning weights. It normally groups filenames by the prefix before the first underscore and prefers a rest/T-pose/idle file; use `--input-list` to choose files explicitly.

For each asset, the output includes `joint_averaged_features.npy` (skin-weighted mean) and `joint_max_weight_features.npy` (dominant-influence variant), both keyed by joint name. The optional checkpoint uses **only** `joint_averaged_features.npy`; `joint_max_weight_features.npy` is not an inference input.

Load the generated `.npy` dictionaries only from a trusted source because they use NumPy pickle objects.
