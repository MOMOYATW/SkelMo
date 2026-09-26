# Joint-feature preprocessing

This module extracts the 768-dimensional rest-pose descriptors used by SkelMo.
It renders a rigged GLB from multiple views, projects mesh vertices into those
views, samples DINO descriptors, and pools the vertex descriptors to joints
using skinning weights.

## Requirements

- Blender 4.x available as `blender`, or passed with `--blender`.
- The SkelMo Python environment from `environment.yaml`.
- Rigged `.glb` files containing a mesh, armature, and skinning weights.

An HDRI is optional. Without one, the renderer uses area lights.

## Run the complete pipeline

From the repository root:

```bash
python -m preprocess.joint_features.pipeline \
  --input-dir /path/to/rigged_glbs \
  --render-root /path/to/work/renders \
  --output-root /path/to/work/features
```

The input directory may contain several animations for the same asset. The
pipeline groups files by the part before the first underscore and selects a
rest, T-pose, idle, or standing file when available. To avoid this naming
convention, provide an explicit list with `--input-list`.

Useful options:

```text
--asset NAME                  Process only selected assets; repeat as needed.
--stages render,project       Run selected stages in canonical order.
--device auto|cuda|mps|cpu    Select the DINO inference device.
--overwrite                   Replace completed outputs.
--dry-run                     Print commands without executing them.
```

The released model expects 768-dimensional features, so the default encoder is
`dinov2_vitb14`. The pipeline rejects incompatible feature dimensions. DINOv3
is also supported when an explicit checkpoint and local source checkout are
provided.

## Pipeline stages

1. `render_rest_pose.py` clears animation, centers the skinned rest pose, and
   renders six axis-aligned plus eight diagonal transparent views.
2. `export_geometry.py`, executed by Blender, exports vertices, skinning
   weights, joint metadata, projected UV coordinates, and visibility.
3. `extract_features.py` samples DINO patch descriptors at visible vertices,
   fills uncovered surface regions, and pools descriptors to joints.
4. `pipeline.py` validates all required files and writes a JSON report.

Each stage is restartable and skips existing complete output unless
`--overwrite` is supplied.

## Outputs

For each asset, `<output-root>/<asset>/` contains:

```text
geometry.npz
geometry_metadata.json
view_features.npz
features.npz
feature_metadata.json
joint_averaged_features.npy
joint_max_weight_features.npy
```

`joint_averaged_features.npy` maps joint names to skin-weighted DINO features
and is consumed by inference. Place it beside the asset's
`processed_skeleton/` directory:

```text
<asset>/
├── joint_averaged_features.npy
└── processed_skeleton/
    └── cond.npy
```

`joint_max_weight_features.npy` pools each vertex under its strongest skinning
influence and is used by the training data loader. `features.npz` retains the
full vertex features, joint features, coverage masks, skinning arrays, and fill
provenance for reproducibility.

## Tests

The unit tests do not require Blender or model weights:

```bash
python -m unittest discover -s preprocess/joint_features/tests -v
```
