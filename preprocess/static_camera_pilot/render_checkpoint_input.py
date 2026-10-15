"""Render an animated GLB and prepare the aligned checkpoint inference input.

Run with Python containing Blender's ``bpy`` module, not the model environment.
The script deliberately exports only frame-zero 3D joints to the inference NPZ;
later posed joints are used solely to determine the same full-clip scene bounds
as the training renderer. The fixed camera and projection are saved for audit.

Rendering conventions follow this repository's static-camera training renderer
(``build_dataset.py``): whole square frames, 30 fps, scene longest side = 1,
35 mm lens / 32 mm sensor, EEVEE studio lights, and no person crop. The compact
rendering code below is adapted from UniMate's MIT-licensed blender_render.py;
see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import (
    normalized_yup_camera,
    pinhole_intrinsics,
    project_world_points,
    transform_points,
    zup_to_yup_matrix,
)


def parent_first_bones(armature, selected_names=None):
    """Return names/parent indices, walking across omitted helper bones."""
    bones = list(armature.data.bones)
    by_name = {bone.name: bone for bone in bones}
    if selected_names is None:
        selected_names = [bone.name for bone in bones]
    if len(selected_names) != len(set(selected_names)) or not set(selected_names) <= set(by_name):
        raise ValueError("Joint list has duplicate or unknown bone names")
    selected = set(selected_names)
    roots = []
    parents_by_name = {}
    for name in selected_names:
        parent = by_name[name].parent
        while parent is not None and parent.name not in selected:
            parent = parent.parent
        parents_by_name[name] = None if parent is None else parent.name
        if parent is None:
            roots.append(name)
    if len(roots) != 1 or not 2 <= len(selected_names) <= 143:
        raise ValueError("Select 2-143 joints with exactly one root")
    index = {name: i for i, name in enumerate(selected_names)}
    parents = np.asarray(
        [-1 if parents_by_name[name] is None else index[parents_by_name[name]]
         for name in selected_names], dtype=np.int64
    )
    if parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(parents))):
        raise ValueError("Joint list must be parent-first, with its root first")
    return selected_names, parents


def camera_matrix(azimuth_deg, distance):
    from mathutils import Vector

    angle = math.radians(azimuth_deg)
    position = Vector((distance * math.cos(angle), distance * math.sin(angle), 0.0))
    rotation = (-position).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    rotation.translation = position
    return rotation


def fit_camera_distance(distance, scene_scale=1.0):
    half_extent = scene_scale / 2.0
    required = half_extent + half_extent * 35.0 / 16.0 * 1.05
    return max(float(distance), required)


def full_clip_bounds(meshes, scene, start_frame, end_frame):
    from mathutils import Vector

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        for mesh in meshes:
            corners = np.asarray([mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box])
            low = np.minimum(low, corners.min(axis=0))
            high = np.maximum(high, corners.max(axis=0))
    if not np.isfinite(low).all() or np.any(high <= low):
        raise ValueError("Could not determine a valid full-clip mesh bounding box")
    return low, high


def joint_heads(armature, names, rest=False):
    from mathutils import Vector

    if rest:
        matrices = [armature.data.bones[name].matrix_local for name in names]
    else:
        matrices = [armature.pose.bones[name].matrix for name in names]
    return np.asarray([
        armature.matrix_world @ (matrix @ Vector((0.0, 0.0, 0.0)))
        for matrix in matrices
    ], dtype=np.float64)


def tied_meshes(armature):
    import bpy

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("GLB has no mesh")

    def belongs(mesh):
        if any(mod.type == "ARMATURE" and mod.object == armature for mod in mesh.modifiers):
            return True
        parent = mesh.parent
        while parent is not None:
            if parent == armature:
                return True
            parent = parent.parent
        return False

    selected = [mesh for mesh in meshes if belongs(mesh)]
    if not selected:
        selected = [max(meshes, key=lambda mesh: len(mesh.data.vertices))]
    for mesh in meshes:
        if mesh not in selected:
            bpy.data.objects.remove(mesh, do_unlink=True)
    return selected


def setup_studio_render(scene, resolution, samples, fps):
    import bpy

    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 1.0
    scene.eevee.gtao_quality = 0.5
    scene.eevee.use_ssr = True
    scene.eevee.use_ssr_refraction = True
    scene.eevee.use_ssr_halfres = True
    scene.eevee.ssr_quality = 0.5
    scene.eevee.ssr_max_roughness = 0.5
    scene.eevee.ssr_thickness = 0.2
    scene.eevee.use_soft_shadows = True
    scene.eevee.shadow_cascade_size = "2048"
    scene.eevee.shadow_cube_size = "1024"
    scene.eevee.use_bloom = True
    scene.render.use_high_quality_normals = True
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.fps = fps
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.7
    output = nodes.new("ShaderNodeOutputWorld")
    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "LINEAR"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = (0.9, 0.9, 0.9, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    for name, rotation, energy, shadow, specular in (
        ("Key_Light", (0.785, 0, -0.785), 1.5, True, 1.0),
        ("Fill_Light", (0.785, 0, 2.356), 0.7, False, 0.5),
        ("Rim_Light", (-0.524, 0, -3.927), 1.0, False, 0.8),
        ("Bottom_Light", (2.618, 0, 0), 0.3, False, 0.3),
    ):
        lamp = bpy.data.lights.new(name, "SUN")
        lamp.energy = energy
        lamp.use_shadow = shadow
        lamp.specular_factor = specular
        if shadow:
            lamp.angle = 0.05
        obj = bpy.data.objects.new(name, lamp)
        bpy.context.collection.objects.link(obj)
        obj.rotation_euler = rotation


def fix_materials(meshes):
    for mesh in meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            tree = material.node_tree
            principled = next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
            if principled is None:
                continue
            metallic = principled.inputs.get("Metallic")
            if metallic is not None and not metallic.is_linked and metallic.default_value > 0.9:
                metallic.default_value = 0.9
            normal = principled.inputs.get("Normal")
            if normal is not None:
                for link in list(normal.links):
                    tree.links.remove(link)
            base = principled.inputs.get("Base Color")
            if base is not None and not base.is_linked:
                color = tuple(base.default_value[:3])
                if all(x > 0.95 for x in color) or all(x < 0.05 for x in color):
                    if mesh.data.vertex_colors:
                        vertex = tree.nodes.new("ShaderNodeVertexColor")
                        vertex.layer_name = mesh.data.vertex_colors[0].name
                        tree.links.new(vertex.outputs["Color"], base)
                    else:
                        image = next((node for node in tree.nodes
                                      if node.type == "TEX_IMAGE" and node.image is not None), None)
                        if image is not None:
                            tree.links.new(image.outputs["Color"], base)
            alpha = principled.inputs.get("Alpha")
            material.blend_method = "HASHED" if alpha is not None and (
                alpha.is_linked or alpha.default_value < 1.0
            ) else "OPAQUE"
            material.show_transparent_back = False
        mesh.data.use_auto_smooth = True
        mesh.data.auto_smooth_angle = math.radians(30)


def save_overlay(frame_path, output, uv, visible, parents):
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for joint, parent in enumerate(parents):
        if parent >= 0 and visible[joint] and visible[parent]:
            draw.line([tuple(uv[parent]), tuple(uv[joint])], fill=(0, 240, 255), width=3)
    for joint, point in enumerate(uv):
        if visible[joint]:
            x, y = map(float, point)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 80, 30))
    image.save(output)


def render(cli):
    import bpy
    from mathutils import Matrix

    if cli.output.exists():
        raise FileExistsError("Output already exists: %s" % cli.output)
    if cli.resolution < 32 or not 1 <= cli.frames <= 32 or cli.fps <= 0:
        raise ValueError("Use resolution >=32, 1-32 frames, and positive fps")
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.render.fps = cli.fps  # glTF animation times are imported at scene FPS.
    bpy.ops.import_scene.gltf(filepath=str(cli.glb.resolve()))
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise ValueError("GLB has no armature")
    armature = max(armatures, key=lambda obj: len(obj.data.bones))
    meshes = tied_meshes(armature)
    if cli.action:
        action = bpy.data.actions.get(cli.action)
        if action is None:
            raise ValueError("Unknown action %s; available: %s" % (cli.action, list(bpy.data.actions)))
    elif armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
    elif len(bpy.data.actions) == 1:
        action = bpy.data.actions[0]
    else:
        raise ValueError("Specify --action; available: %s" % list(bpy.data.actions))
    armature.animation_data_create()
    armature.animation_data.action = action
    start = int(math.ceil(action.frame_range[0])) if cli.start_frame is None else cli.start_frame
    end = start + cli.frames - 1
    if end > int(math.floor(action.frame_range[1])):
        raise ValueError("Requested window exceeds action frame range %s" % tuple(action.frame_range))
    normalization_start = int(math.ceil(action.frame_range[0]))
    normalization_end = (
        int(math.floor(action.frame_range[1]))
        if cli.normalization_end_frame is None else cli.normalization_end_frame
    )
    if normalization_end < end or normalization_end > int(math.floor(action.frame_range[1])):
        raise ValueError("Normalization range must include rendered frames and stay inside the action")
    scene.frame_start, scene.frame_end = start, end
    names = None if cli.joints_json is None else json.loads(cli.joints_json.read_text())
    if names is not None and not isinstance(names, list):
        raise ValueError("--joints-json must contain a JSON list of bone names")
    names, parents = parent_first_bones(armature, names)
    scene.frame_set(start)
    pose_source_zup = joint_heads(armature, names)
    rest_source_zup = joint_heads(armature, names, rest=True)
    lower, upper = full_clip_bounds(meshes, scene, normalization_start, normalization_end)
    scale = 1.0 / float(np.max(upper - lower))
    offset = -0.5 * (lower + upper)

    node = bpy.data.objects.new("NormalizationNode", None)
    scene.collection.objects.link(node)
    for obj in list(bpy.data.objects):
        if obj is not node and obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}:
            world = obj.matrix_world.copy()
            obj.parent = node
            obj.matrix_world = world
    node.scale = (scale,) * 3
    node.location = tuple(offset * scale)
    bpy.context.view_layer.update()
    setup_studio_render(scene, cli.resolution, cli.samples, cli.fps)
    fix_materials(meshes)
    camera = bpy.data.cameras.new("CheckpointCamera")
    camera.type = "PERSP"
    camera.lens = 35.0
    camera.sensor_width = 32.0
    camera_obj = bpy.data.objects.new("CheckpointCamera", camera)
    scene.collection.objects.link(camera_obj)
    camera_obj.matrix_world = camera_matrix(cli.azimuth, fit_camera_distance(cli.camera_distance))
    scene.camera = camera_obj
    bpy.context.view_layer.update()

    transforms = normalized_yup_camera(np.asarray(camera_obj.matrix_world), scale, offset)
    intrinsics = pinhole_intrinsics(cli.resolution, camera.angle_x)
    axis = zup_to_yup_matrix()
    pose_yup = pose_source_zup @ axis.T
    rest_yup = rest_source_zup @ axis.T
    first_world = transform_points(transforms["source_yup_to_world"], pose_yup)
    rest_world = transform_points(transforms["source_yup_to_world"], rest_yup)
    first_pose = transform_points(transforms["world_to_camera_gravity"], first_world)
    rest_pose = transform_points(transforms["world_to_camera_gravity"], rest_world)
    uv, depth, visible = project_world_points(
        first_world, transforms["world_to_camera_cv"], intrinsics, cli.resolution
    )
    if not visible.any():
        raise ValueError("No first-frame joints project into the rendered image")

    cli.output.mkdir(parents=True)
    frame_dir = cli.output / "frames"
    frame_dir.mkdir()
    scene.render.image_settings.file_format = cli.format.upper()
    scene.render.image_settings.color_mode = "RGBA"
    if cli.format == "webp":
        scene.render.image_settings.quality = 90
    for index, frame in enumerate(range(start, end + 1)):
        scene.frame_set(frame)
        frame_path = frame_dir / ("%04d.%s" % (index, cli.format))
        scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        if not frame_path.is_file():
            raise RuntimeError("Render did not produce %s" % frame_path)
    np.savez_compressed(
        cli.output / "firstframe_input.npz",
        parents=parents,
        joint_names=np.asarray(names),
        first_pose=first_pose.astype(np.float32),
        rest_pose=rest_pose.astype(np.float32),
        joint_uv=uv.astype(np.float32),
        joint_visible=visible,
        resolution=np.asarray(cli.resolution, dtype=np.int32),
        fps=np.asarray(cli.fps, dtype=np.int32),
    )
    save_overlay(frame_dir / ("0000.%s" % cli.format),
                 cli.output / "first_frame_projection.png", uv, visible, parents)
    meta = {
        "format": "skelmo_checkpoint_render_v1",
        "source_glb": str(cli.glb.resolve()),
        "action": action.name,
        "frame_start": start,
        "frame_end": end,
        "normalization_frame_start": normalization_start,
        "normalization_frame_end": normalization_end,
        "fps": cli.fps,
        "resolution": cli.resolution,
        "azimuth_deg": cli.azimuth,
        "camera_distance": fit_camera_distance(cli.camera_distance),
        "lens_mm": 35.0,
        "sensor_width_mm": 32.0,
        "source_bbox_lower_zup": lower.tolist(),
        "source_bbox_upper_zup": upper.tolist(),
        "scene_scale": scale,
        "scene_offset_zup": offset.tolist(),
        "camera_to_world_zup": np.asarray(camera_obj.matrix_world).tolist(),
        "source_yup_to_world": transforms["source_yup_to_world"].tolist(),
        "world_to_camera_gravity": transforms["world_to_camera_gravity"].tolist(),
        "world_to_camera_cv": transforms["world_to_camera_cv"].tolist(),
        "intrinsics": intrinsics.tolist(),
        "first_depth_cv": depth.tolist(),
    }
    (cli.output / "camera.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"output": str(cli.output), "frames": cli.frames,
                      "joints": len(parents), "visible_joints": int(visible.sum())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, type=Path, help="Animated, rigged GLB")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--action", help="Blender action name; required when ambiguous")
    parser.add_argument("--joints-json", type=Path, help="Parent-first JSON list of bone names")
    parser.add_argument("--start-frame", type=int, help="First source animation frame")
    parser.add_argument("--normalization-end-frame", type=int,
                        help="Last action frame used to fit scene scale (default: action end)")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--azimuth", type=float, default=-90.0, help="Fixed camera yaw in degrees")
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--format", choices=("png", "webp"), default="webp")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
