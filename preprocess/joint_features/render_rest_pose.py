"""Export a skinned rest-pose GLB and render transparent multi-view images."""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import bpy
import numpy as np
from mathutils import Vector

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from views import DEFAULT_VIEW_NAMES, VIEW_DIRECTIONS, parse_view_names

DEFAULT_HDRI_PATH = os.path.join(os.path.dirname(__file__), "skarpa-winter-forest_4K.exr")
PREFERRED_REPRESENTATIVE_KEYWORDS = ("tpose", "t-pose", "rest", "idle", "stand")
REST_POSE_GLB_NAME = "rest_pose.glb"


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for datablock_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.actions,
    ):
        for datablock in list(datablock_collection):
            datablock_collection.remove(datablock, do_unlink=True)


def load_model(path: str) -> None:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
    else:
        raise ValueError(f"Unsupported model format: {suffix or path}")
    bpy.context.view_layer.update()


def get_meshes() -> Iterable[bpy.types.Object]:
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and not obj.hide_render:
            yield obj


def clear_animation_to_rest_pose() -> None:
    scene = bpy.context.scene
    scene.frame_set(0)

    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()

        if obj.type == "ARMATURE":
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.select_all(action="SELECT")
            bpy.ops.pose.transforms_clear()
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.select_set(False)

    bpy.context.view_layer.update()


def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False

    for obj in get_meshes():
        found = True
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            bbox_min.x = min(bbox_min.x, world_corner.x)
            bbox_min.y = min(bbox_min.y, world_corner.y)
            bbox_min.z = min(bbox_min.z, world_corner.z)
            bbox_max.x = max(bbox_max.x, world_corner.x)
            bbox_max.y = max(bbox_max.y, world_corner.y)
            bbox_max.z = max(bbox_max.z, world_corner.z)

    if not found:
        raise RuntimeError("No visible mesh objects found in scene.")

    return bbox_min, bbox_max


def center_scene() -> Tuple[Vector, Vector]:
    bbox_min, bbox_max = scene_bbox()
    center = (bbox_min + bbox_max) / 2.0

    for obj in bpy.context.scene.objects:
        if obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}:
            obj.matrix_world.translation -= center

    bpy.context.view_layer.update()
    return scene_bbox()


def export_rest_pose_glb(output_path: str) -> None:
    """Export the current rest pose while preserving the skinned mesh and rig."""
    bpy.ops.object.select_all(action="DESELECT")
    export_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type not in {"CAMERA", "LIGHT"}
    ]
    for obj in export_objects:
        obj.select_set(True)

    if not any(obj.type == "MESH" for obj in export_objects):
        raise RuntimeError("Cannot export rest-pose GLB: no mesh objects found.")
    if not any(obj.type == "ARMATURE" for obj in export_objects):
        raise RuntimeError("Cannot export rest-pose GLB: no armature found.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_kwargs = {
        "filepath": output_path,
        "export_format": "GLB",
        "use_selection": True,
        "export_cameras": False,
        "export_lights": False,
        "export_animations": False,
        "export_skins": True,
        "export_all_influences": True,
        # Applying modifiers can bake away the armature deformation relationship.
        "export_apply": False,
    }
    supported = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    bpy.ops.export_scene.gltf(
        **{key: value for key, value in export_kwargs.items() if key in supported}
    )


def prepare_rest_pose(object_path: str) -> Tuple[Vector, Vector]:
    load_model(object_path)
    clear_animation_to_rest_pose()
    return center_scene()


def save_rest_pose_glb(object_path: str, output_dir: str) -> str:
    """Load one source GLB and save its centered, animation-free rest pose."""
    reset_scene()
    prepare_rest_pose(object_path)
    output_path = os.path.join(output_dir, REST_POSE_GLB_NAME)
    export_rest_pose_glb(output_path)
    return output_path


def add_rest_pose_glb_to_metadata(metadata_path: str) -> None:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["rest_pose_glb"] = REST_POSE_GLB_NAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def camera_distance(bbox_min: Vector, bbox_max: Vector, margin: float) -> float:
    extents = bbox_max - bbox_min
    radius = max(extents.x, extents.y, extents.z) / 2.0
    camera = bpy.context.scene.camera
    fov = camera.data.angle if camera and camera.data else math.radians(50.0)
    return max(radius / math.tan(fov / 2.0) * margin, 0.01)


def setup_camera() -> bpy.types.Object:
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.lens = 70
    camera.data.type = "PERSP"
    camera.data.clip_start = 0.001
    bpy.context.scene.camera = camera
    return camera


def look_at(
    camera: bpy.types.Object,
    direction: Vector,
    distance: float,
    target: Vector | None = None,
) -> None:
    target = Vector((0.0, 0.0, 0.0)) if target is None else target
    camera.location = target + direction.normalized() * distance
    look_direction = target - camera.location
    camera.rotation_euler = look_direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def preview_alpha_bounds(resolution: int = 96) -> Tuple[float, float, float, float]:
    scene = bpy.context.scene
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    original_filepath = scene.render.filepath
    original_cycles_samples = scene.cycles.samples
    try:
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = 1
        with tempfile.TemporaryDirectory(prefix="skelmo-rest-alpha-preview-") as temp_dir:
            preview_path = Path(temp_dir) / "preview.png"
            scene.render.filepath = str(preview_path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(preview_path), check_existing=False)
            try:
                pixels = np.empty(len(image.pixels), dtype=np.float32)
                image.pixels.foreach_get(pixels)
                alpha = pixels.reshape(resolution, resolution, 4)[..., 3]
            finally:
                bpy.data.images.remove(image)
    finally:
        scene.render.resolution_x, scene.render.resolution_y = original_resolution
        scene.render.filepath = original_filepath
        scene.cycles.samples = original_cycles_samples

    ys, xs = np.nonzero(alpha > 0.01)
    if not len(xs):
        raise RuntimeError("Transparent framing preview did not contain any visible pixels")
    return (
        float(xs.min()) / resolution,
        float(ys.min()) / resolution,
        float(xs.max() + 1) / resolution,
        float(ys.max() + 1) / resolution,
    )


def calibrate_view(
    camera: bpy.types.Object,
    direction: Vector,
    distance: float,
    margin: float,
    iterations: int = 2,
) -> Tuple[float, Vector, Tuple[float, float, float, float]]:
    """Tightly frame the pixels actually visible through material alpha."""
    target = Vector((0.0, 0.0, 0.0))
    bounds = (0.0, 0.0, 1.0, 1.0)
    look_at(camera, direction, distance, target)
    for _ in range(iterations):
        bounds = preview_alpha_bounds()
        u_min, v_min, u_max, v_max = bounds
        center_u = (u_min + u_max) / 2.0
        center_v = (v_min + v_max) / 2.0
        tan_half_x = math.tan(camera.data.angle_x / 2.0)
        scene = bpy.context.scene
        aspect = (
            scene.render.resolution_x * scene.render.pixel_aspect_x
        ) / max(scene.render.resolution_y * scene.render.pixel_aspect_y, 1.0e-8)
        tan_half_y = tan_half_x / aspect
        rotation = camera.matrix_world.to_quaternion()
        right = rotation @ Vector((1.0, 0.0, 0.0))
        up = rotation @ Vector((0.0, 1.0, 0.0))
        target += right * ((center_u - 0.5) * 2.0 * distance * tan_half_x)
        target += up * ((center_v - 0.5) * 2.0 * distance * tan_half_y)
        occupied = max(u_max - u_min, v_max - v_min)
        distance = max(distance * occupied * margin, 0.01)
        look_at(camera, direction, distance, target)
    return distance, target, bounds


def setup_hdri(hdri_path: str, strength: float) -> bool:
    if not hdri_path or not os.path.isfile(hdri_path):
        return False

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    background = nodes.new(type="ShaderNodeBackground")
    environment = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")

    environment.image = bpy.data.images.load(hdri_path)
    background.inputs["Strength"].default_value = strength
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return True


def setup_area_lights() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = False
    world.color = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.0))
    key = bpy.context.object
    key.name = "Key_Area"
    key.data.energy = 500
    key.data.size = 5

    bpy.ops.object.light_add(type="AREA", location=(0.0, 3.0, 3.0))
    fill = bpy.context.object
    fill.name = "Fill_Area"
    fill.data.energy = 120
    fill.data.size = 6


def setup_render(resolution: int, engine: str, samples: int) -> None:
    scene = bpy.context.scene
    render = scene.render
    render.engine = "BLENDER_EEVEE_NEXT" if engine == "BLENDER_EEVEE" else engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.film_transparent = True
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100

    if render.engine == "CYCLES":
        configure_cycles_gpu()
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.transparent_max_bounces = 3
        try:
            scene.cycles.device = "GPU"
        except Exception:
            pass
    else:
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = samples


def write_metadata(
    path: str,
    source_path: str,
    bbox_min: Vector,
    bbox_max: Vector,
    distance: float,
    hdri_path: str,
    hdri_strength: float,
    view_cameras: dict,
    engine: str,
    samples: int,
    resolution: int,
    species: str = None,
    view_names: List[str] = None,
) -> None:
    camera = bpy.context.scene.camera
    data = {
        "source_path": source_path,
        "rest_pose_glb": REST_POSE_GLB_NAME,
        "views": list(view_names or DEFAULT_VIEW_NAMES),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "camera_distance": distance,
        "camera_angle_x": camera.data.angle_x,
        "camera_angle_y": camera.data.angle_y,
        "view_cameras": view_cameras,
        "hdri_path": hdri_path if os.path.isfile(hdri_path) else None,
        "hdri_strength": hdri_strength if os.path.isfile(hdri_path) else None,
        "engine": bpy.context.scene.render.engine,
        "samples": samples,
        "resolution": resolution,
    }
    if species is not None:
        data["species"] = species
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def render_views(
    object_path: str,
    output_dir: str,
    resolution: int,
    engine: str,
    samples: int,
    margin: float,
    hdri_path: str,
    hdri_strength: float,
    species: str = None,
    skip_existing_views: bool = False,
    view_names: List[str] = None,
) -> None:
    reset_scene()
    setup_render(resolution, engine, samples)
    setup_camera()
    if setup_hdri(hdri_path, hdri_strength):
        print(f"Using HDRI lighting: {hdri_path} (strength={hdri_strength})")
    else:
        print("HDRI not found; using fallback area lights.")
        setup_area_lights()

    os.makedirs(output_dir, exist_ok=True)
    bbox_min, bbox_max = prepare_rest_pose(object_path)
    export_rest_pose_glb(os.path.join(output_dir, REST_POSE_GLB_NAME))
    distance = camera_distance(bbox_min, bbox_max, margin)

    metadata_path = Path(output_dir) / "metadata.json"
    previous_metadata = {}
    if metadata_path.is_file():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    view_cameras = dict(previous_metadata.get("view_cameras", {}))
    selected_views = list(view_names or DEFAULT_VIEW_NAMES)
    for view_name in selected_views:
        direction_tuple = VIEW_DIRECTIONS[view_name]
        output_path = os.path.join(output_dir, f"{view_name}.png")
        if (
            skip_existing_views
            and os.path.isfile(output_path)
            and view_name in view_cameras
            and "matrix_world" in view_cameras[view_name]
        ):
            continue
        direction = Vector(direction_tuple)
        view_distance, target, alpha_bounds = calibrate_view(
            bpy.context.scene.camera, direction, distance, margin
        )
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        camera = bpy.context.scene.camera
        view_cameras[view_name] = {
            "distance": view_distance,
            "target": list(target),
            "alpha_bounds": list(alpha_bounds),
            "matrix_world": [list(row) for row in camera.matrix_world],
        }

    write_metadata(
        os.path.join(output_dir, "metadata.json"),
        object_path,
        bbox_min,
        bbox_max,
        distance,
        hdri_path,
        hdri_strength,
        view_cameras,
        engine,
        samples,
        resolution,
        species,
        selected_views,
    )


def species_from_glb_path(path: str) -> str:
    if os.path.splitext(path)[1].lower() == ".fbx":
        return os.path.basename(os.path.dirname(path))
    return Path(path).stem.split("_", 1)[0]


def representative_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path).lower()
    for index, keyword in enumerate(PREFERRED_REPRESENTATIVE_KEYWORDS):
        if keyword in name:
            return index, name
    return len(PREFERRED_REPRESENTATIVE_KEYWORDS), name


def configure_cycles_gpu() -> str:
    """Select the first available GPU backend, honoring CYCLES_COMPUTE_DEVICE."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    requested = os.environ.get("CYCLES_COMPUTE_DEVICE", "").strip().upper()
    backends = [requested] if requested else ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
    for backend in backends:
        if not backend:
            continue
        try:
            preferences.compute_device_type = backend
            preferences.refresh_devices()
        except Exception:
            continue
        gpu_devices = [device for device in preferences.devices if device.type != "CPU"]
        if not gpu_devices:
            continue
        for device in preferences.devices:
            device.use = device.type != "CPU"
        print(f"Cycles backend: {backend}; devices: {[d.name for d in gpu_devices]}")
        return backend
    print("WARNING: no Cycles GPU backend found; rendering on CPU")
    return "CPU"


def select_species_representatives(
    input_dir: str,
    input_list: str = None,
    species_filter: set[str] | None = None,
) -> List[Tuple[str, str]]:
    grouped: Dict[str, List[str]] = {}
    if input_list:
        paths = []
        with open(input_list, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line)
                paths.append(path if path.is_absolute() else Path(input_dir) / path)
    else:
        paths = [Path(input_dir) / name for name in sorted(os.listdir(input_dir))]
    for path in paths:
        if path.suffix.lower() != ".glb":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Listed GLB does not exist: {path}")
        path_string = str(path)
        grouped.setdefault(species_from_glb_path(path_string), []).append(path_string)

    representatives = []
    for species, paths in sorted(grouped.items()):
        if species_filter and species not in species_filter:
            continue
        representatives.append((species, sorted(paths, key=representative_sort_key)[0]))
    return representatives


def render_species_batch(
    input_dir: str,
    output_root: str,
    resolution: int,
    engine: str,
    samples: int,
    margin: float,
    hdri_path: str,
    hdri_strength: float,
    overwrite: bool,
    input_list: str = None,
    report_path: str = None,
    view_names: List[str] = None,
    species_filter: set[str] | None = None,
) -> None:
    representatives = select_species_representatives(
        input_dir, input_list, species_filter
    )
    os.makedirs(output_root, exist_ok=True)
    report_path = report_path or os.path.join(output_root, "_batch_report.json")

    report = {
        "input_dir": input_dir,
        "output_root": output_root,
        "input_list": input_list,
        "total_species": len(representatives),
        "engine": engine,
        "samples": samples,
        "resolution": resolution,
        "rendered": [],
        "skipped": [],
        "errors": [],
    }

    for index, (species, object_path) in enumerate(representatives, start=1):
        output_dir = os.path.join(output_root, species)
        metadata_path = os.path.join(output_dir, "metadata.json")
        rest_pose_glb_path = os.path.join(output_dir, REST_POSE_GLB_NAME)
        selected_views = list(view_names or DEFAULT_VIEW_NAMES)
        complete_views = all(
            os.path.isfile(os.path.join(output_dir, f"{view_name}.png"))
            for view_name in selected_views
        )
        if (
            os.path.isfile(metadata_path)
            and os.path.isfile(rest_pose_glb_path)
            and complete_views
            and not overwrite
        ):
            print(f"[{index}/{len(representatives)}] skip existing: {species}")
            report["skipped"].append({"species": species, "object_path": object_path, "output_dir": output_dir})
            continue

        if (
            os.path.isfile(metadata_path)
            and not os.path.isfile(rest_pose_glb_path)
            and not overwrite
        ):
            print(f"[{index}/{len(representatives)}] export rest-pose GLB: {species}")
            try:
                exported_glb = save_rest_pose_glb(object_path, output_dir)
                add_rest_pose_glb_to_metadata(metadata_path)
                report["rendered"].append(
                    {
                        "species": species,
                        "object_path": object_path,
                        "output_dir": output_dir,
                        "rest_pose_glb": exported_glb,
                        "rendered_images": False,
                    }
                )
            except Exception as exc:
                print(f"ERROR exporting {species}: {exc}")
                report["errors"].append({"species": species, "object_path": object_path, "error": str(exc)})
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
            continue

        print(f"[{index}/{len(representatives)}] render {species}: {os.path.basename(object_path)}")
        try:
            render_views(
                object_path,
                output_dir,
                resolution,
                engine,
                samples,
                margin,
                hdri_path,
                hdri_strength,
                species,
                skip_existing_views=not overwrite,
                view_names=selected_views,
            )
            report["rendered"].append(
                {
                    "species": species,
                    "object_path": object_path,
                    "output_dir": output_dir,
                    "rest_pose_glb": rest_pose_glb_path,
                    "rendered_images": True,
                }
            )
        except Exception as exc:
            print(f"ERROR rendering {species}: {exc}")
            report["errors"].append({"species": species, "object_path": object_path, "error": str(exc)})

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(
        f"Batch complete: rendered={len(report['rendered'])}, "
        f"skipped={len(report['skipped'])}, errors={len(report['errors'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-path", "--object_path", dest="object_path")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir")
    parser.add_argument("--output-root", "--output_root", dest="output_root")
    parser.add_argument(
        "--one-per-species",
        "--one_per_species",
        dest="one_per_species",
        action="store_true",
    )
    parser.add_argument(
        "--asset", "--animal", dest="animal", metavar="NAME", action="append", default=[]
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-list")
    parser.add_argument("--report")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], default="CYCLES")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument(
        "--hdri-path", "--hdri_path", dest="hdri_path", default=DEFAULT_HDRI_PATH
    )
    parser.add_argument(
        "--hdri-strength",
        "--hdri_strength",
        dest="hdri_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEW_NAMES),
        help="Comma-separated views to render (for example: front).",
    )

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    try:
        view_names = parse_view_names(args.views)
    except ValueError as exc:
        parser.error(str(exc))

    if args.one_per_species:
        if not args.input_dir or not args.output_root:
            parser.error("--one_per_species requires --input_dir and --output_root")
        render_species_batch(
            args.input_dir,
            args.output_root,
            args.resolution,
            args.engine,
            args.samples,
            args.margin,
            args.hdri_path,
            args.hdri_strength,
            args.overwrite,
            args.input_list,
            args.report,
            view_names,
            set(args.animal),
        )
        return

    if not args.object_path or not args.output_dir:
        parser.error("single-model mode requires --object_path and --output_dir")

    render_views(
        args.object_path,
        args.output_dir,
        args.resolution,
        args.engine,
        args.samples,
        args.margin,
        args.hdri_path,
        args.hdri_strength,
        species_from_glb_path(args.object_path),
        view_names=view_names,
    )


if __name__ == "__main__":
    main()
