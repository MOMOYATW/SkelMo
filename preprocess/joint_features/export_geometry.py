"""Export rest-pose vertices, skinning weights, and multi-view visibility.

Run this script with Blender.  The resulting ``geometry.npz`` files are the
geometry half of the SkelMo DINO feature pipeline; they intentionally do
not depend on PyTorch and can be regenerated independently of the image
encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from views import DEFAULT_VIEW_NAMES, VIEW_DIRECTIONS, parse_view_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-root",
        type=Path,
        default=REPO_ROOT / "data/joint_features/renders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/joint_features/features",
    )
    parser.add_argument(
        "--asset", "--animal", dest="animal", metavar="NAME", action="append", default=[]
    )
    parser.add_argument("--max-influences", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.objects,
    ):
        for datablock in list(collection):
            collection.remove(datablock, do_unlink=True)


def import_rest_pose(path: Path) -> None:
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True)
    bpy.context.scene.frame_set(0)
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()


def primary_armature():
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature in rest_pose.glb")
    return max(armatures, key=lambda obj: len(obj.data.bones))


def visible_meshes():
    return sorted(
        [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render],
        key=lambda obj: obj.name,
    )


def extract_geometry(max_influences: int) -> dict:
    armature = primary_armature()
    meshes = visible_meshes()
    if not meshes:
        raise RuntimeError("No visible mesh in rest_pose.glb")

    bones = list(armature.data.bones)
    joint_names = [bone.name for bone in bones]
    joint_lookup = {name: index for index, name in enumerate(joint_names)}
    joint_parents = np.asarray(
        [joint_lookup.get(bone.parent.name, -1) if bone.parent else -1 for bone in bones],
        dtype=np.int32,
    )
    joint_positions = np.asarray(
        [armature.matrix_world @ bone.head_local for bone in bones], dtype=np.float32
    )

    all_vertices = []
    all_normals = []
    all_triangles = []
    all_mesh_ids = []
    influence_lists = []
    vertex_offset = 0

    for mesh_id, obj in enumerate(meshes):
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        world_normals = []
        for vertex in obj.data.vertices:
            normal = normal_matrix @ vertex.normal
            normal.normalize()
            world_normals.append(tuple(normal))

        obj.data.calc_loop_triangles()
        all_triangles.extend(
            tuple(vertex_offset + index for index in triangle.vertices)
            for triangle in obj.data.loop_triangles
        )
        all_vertices.extend(tuple(vertex) for vertex in world_vertices)
        all_normals.extend(world_normals)
        all_mesh_ids.extend([mesh_id] * len(world_vertices))

        group_names = {group.index: group.name for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            influences = []
            for assignment in vertex.groups:
                joint_index = joint_lookup.get(group_names.get(assignment.group, ""))
                if joint_index is not None and assignment.weight > 0.0:
                    influences.append((joint_index, float(assignment.weight)))
            influences.sort(key=lambda item: item[1], reverse=True)
            influence_lists.append(influences[:max_influences])
        vertex_offset += len(world_vertices)

    vertices = np.asarray(all_vertices, dtype=np.float32)
    normals = np.asarray(all_normals, dtype=np.float32)
    triangles = np.asarray(all_triangles, dtype=np.int32)
    mesh_ids = np.asarray(all_mesh_ids, dtype=np.int32)
    skin_indices = np.full((len(vertices), max_influences), -1, dtype=np.int32)
    skin_weights = np.zeros((len(vertices), max_influences), dtype=np.float32)
    for vertex_index, influences in enumerate(influence_lists):
        if not influences:
            continue
        total = sum(weight for _, weight in influences)
        if total <= 0.0:
            continue
        for slot, (joint_index, weight) in enumerate(influences):
            skin_indices[vertex_index, slot] = joint_index
            skin_weights[vertex_index, slot] = weight / total

    return {
        "vertices": vertices,
        "normals": normals,
        "triangles": triangles,
        "mesh_ids": mesh_ids,
        "mesh_names": np.asarray([obj.name for obj in meshes]),
        "joint_names": np.asarray(joint_names),
        "joint_parents": joint_parents,
        "joint_positions": joint_positions,
        "skin_joint_indices": skin_indices,
        "skin_weights": skin_weights,
    }


def setup_camera(resolution: int):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "ProjectionCamera"
    camera.data.lens = 70
    camera.data.type = "PERSP"
    camera.data.clip_start = 0.001
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    return camera


def build_bvh(vertices: np.ndarray, triangles: np.ndarray) -> BVHTree:
    return BVHTree.FromPolygons(
        [Vector(vertex) for vertex in vertices],
        [tuple(int(index) for index in triangle) for triangle in triangles],
        all_triangles=True,
    )


def project_views(geometry: dict, metadata: dict, resolution: int) -> dict:
    scene = bpy.context.scene
    camera = setup_camera(resolution)
    distance = float(metadata["camera_distance"])
    bvh = build_bvh(geometry["vertices"], geometry["triangles"])
    vertices = geometry["vertices"]
    normals = geometry["normals"]
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tolerance = max(2.0e-5 * bbox_diag, 1.0e-6)

    view_names = parse_view_names(metadata.get("views", DEFAULT_VIEW_NAMES))
    uv = np.zeros((len(view_names), len(vertices), 2), dtype=np.float32)
    visible = np.zeros((len(view_names), len(vertices)), dtype=bool)
    in_frame = np.zeros((len(view_names), len(vertices)), dtype=bool)
    view_weight = np.zeros((len(view_names), len(vertices)), dtype=np.float32)
    camera_positions = np.zeros((len(view_names), 3), dtype=np.float32)

    for view_index, view_name in enumerate(view_names):
        camera_config = metadata.get("view_cameras", {}).get(view_name)
        if camera_config and "matrix_world" in camera_config:
            camera.matrix_world = Matrix(camera_config["matrix_world"])
        else:
            direction = Vector(VIEW_DIRECTIONS[view_name])
            camera.location = direction.normalized() * distance
            camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        camera_position = np.asarray(camera.location, dtype=np.float32)
        camera_positions[view_index] = camera_position

        for vertex_index, vertex_np in enumerate(vertices):
            vertex = Vector(vertex_np)
            projected = world_to_camera_view(scene, camera, vertex)
            u = float(projected.x)
            v = 1.0 - float(projected.y)
            uv[view_index, vertex_index] = (u, v)
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and projected.z > 0.0):
                continue

            in_frame[view_index, vertex_index] = True
            to_camera = camera_position - vertex_np
            to_camera /= max(float(np.linalg.norm(to_camera)), 1.0e-12)
            # Absolute cosine supports intentionally two-sided wing/ear meshes.
            view_weight[view_index, vertex_index] = max(
                abs(float(np.dot(normals[vertex_index], to_camera))), 0.05
            )

            to_camera_vec = camera.location - vertex
            target_distance = to_camera_vec.length
            if target_distance <= 0.0:
                continue
            to_camera_vec.normalize()
            # Starting just outside the target vertex avoids the numerically
            # ambiguous "ray hits a shared vertex" case.  A visible surface
            # has an unobstructed segment from here to the camera; a hidden
            # surface intersects another triangle first.
            ray_origin = vertex + to_camera_vec * (4.0 * tolerance)
            hit = bvh.ray_cast(ray_origin, to_camera_vec, target_distance)
            hit_distance = hit[3]
            # BVH may immediately re-hit one of the triangles sharing the
            # queried vertex (especially on thin insect legs and wings).  A
            # near-zero hit is numerical self-intersection, not occlusion.
            if hit_distance is not None and float(hit_distance) > 16.0 * tolerance:
                continue

            visible[view_index, vertex_index] = True

    return {
        "view_names": np.asarray(view_names),
        "vertex_uv": uv,
        "vertex_in_frame": in_frame,
        "vertex_visible": visible,
        "vertex_view_weight": view_weight,
        "camera_positions": camera_positions,
        "resolution": np.asarray(resolution, dtype=np.int32),
        "visibility_tolerance": np.asarray(tolerance, dtype=np.float32),
    }


def export_animal(animal_dir: Path, output_dir: Path, max_influences: int) -> dict:
    metadata_path = animal_dir / "metadata.json"
    glb_path = animal_dir / "rest_pose.glb"
    if not metadata_path.is_file() or not glb_path.is_file():
        raise FileNotFoundError(f"Missing rest pose inputs in {animal_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resolution = 512
    front_path = animal_dir / "front.png"
    if front_path.is_file():
        image = bpy.data.images.load(str(front_path), check_existing=False)
        resolution = int(image.size[0])
        bpy.data.images.remove(image)

    import_rest_pose(glb_path)
    geometry = extract_geometry(max_influences)
    projections = project_views(geometry, metadata, resolution)
    arrays = {**geometry, **projections}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "geometry.npz", **arrays)

    direct = projections["vertex_visible"].any(axis=0)
    skinned = geometry["skin_weights"].sum(axis=1) > 0
    result = {
        "animal": animal_dir.name,
        "vertices": int(len(geometry["vertices"])),
        "triangles": int(len(geometry["triangles"])),
        "joints": int(len(geometry["joint_names"])),
        "meshes": int(len(geometry["mesh_names"])),
        "directly_visible_vertices": int(direct.sum()),
        "direct_coverage": float(direct.mean()),
        "skinned_vertices": int(skinned.sum()),
        "unskinned_vertices": int((~skinned).sum()),
        "geometry": str(output_dir / "geometry.npz"),
    }
    (output_dir / "geometry_metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    render_root = args.render_root.resolve()
    output_root = args.output_root.resolve()
    selected = set(args.animal)
    animal_dirs = sorted(
        path
        for path in render_root.iterdir()
        if path.is_dir() and (not selected or path.name in selected)
    )
    report = {"render_root": str(render_root), "output_root": str(output_root), "results": [], "errors": []}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, animal_dir in enumerate(animal_dirs, 1):
        output_dir = output_root / animal_dir.name
        geometry_path = output_dir / "geometry.npz"
        if geometry_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(animal_dirs)}] skip {animal_dir.name}")
            continue
        print(f"[{index}/{len(animal_dirs)}] project {animal_dir.name}")
        try:
            result = export_animal(animal_dir, output_dir, args.max_influences)
            report["results"].append(result)
            print(
                f"  vertices={result['vertices']} joints={result['joints']} "
                f"coverage={result['direct_coverage']:.2%}"
            )
        except Exception as exc:
            report["errors"].append(
                {"animal": animal_dir.name, "error": str(exc), "traceback": traceback.format_exc()}
            )
            print(f"ERROR {animal_dir.name}: {exc}")
        (output_root / "_geometry_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"Done: exported={len(report['results'])}, errors={len(report['errors'])}")


if __name__ == "__main__":
    main()
