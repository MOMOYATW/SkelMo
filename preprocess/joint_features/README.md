# Joint-feature preprocessing

This module extracts the 768-dimensional rest-pose descriptors used by the
optional rest-pose appearance checkpoint. The primary checkpoint does not need
these descriptors.
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

The optional rest-pose appearance checkpoint expects features from
`dinov2_vitb14`, the default encoder. The pipeline rejects incompatible feature
dimensions. Its DINOv3 option is for separate preprocessing experiments, not
for this checkpoint; matching dimensionality alone is not sufficient.

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

Both joint-feature files map joint names to DINO descriptors.
`joint_max_weight_features.npy` pools each vertex under its strongest skinning
influence; `joint_averaged_features.npy` uses skin-weighted averaging. For the
optional checkpoint, match the names to the prepared skeleton's joint order,
take the max-weight descriptor when present and fall back to the averaged one,
then stack them as `rest_appearance` with shape `[J,768]`. Also provide
`rest_pose` with shape `[J,3]`, transformed into the same camera frame and scale
as `first_pose`. See the collapsed rest-pose section of the root README for the
inference command. Load these `.npy` dictionaries only when you trust their
source. `features.npz` retains the full vertex features, joint features,
coverage masks, skinning arrays, and fill provenance for reproducibility.

## Tests

The unit tests do not require Blender or model weights:

```bash
python -m unittest discover -s preprocess/joint_features/tests -v
```
