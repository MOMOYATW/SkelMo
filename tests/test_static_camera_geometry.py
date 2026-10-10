import unittest

import numpy as np

from preprocess.static_camera_pilot.build_dataset import (
    _select_action,
    deterministic_camera_azimuth,
)
from preprocess.static_camera_pilot.geometry import (
    matrix_to_rotation_6d_columns,
    normalized_yup_camera,
    pinhole_intrinsics,
    project_world_points,
    project_to_so3,
    quaternion_wxyz_to_matrix,
    skinning_rotations,
    skinning_rotations_from_matrices,
    transform_points,
)


class StaticCameraGeometryTests(unittest.TestCase):
    def test_source_normalization_and_camera_round_trip(self):
        c2w_zup = np.eye(4)
        # Blender front camera: located on -Y, local -Z looks toward the
        # origin, and local +Y follows Z-up gravity.
        c2w_zup[:3, :3] = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
        c2w_zup[:3, 3] = [0.0, -3.0, 0.0]
        transforms = normalized_yup_camera(c2w_zup, 2.0, [0.5, 1.0, -2.0])
        points = np.asarray([[0.1, 0.2, 0.3], [-1.0, 2.0, 4.0]])
        world = transform_points(transforms["source_yup_to_world"], points)
        recovered = transform_points(
            np.linalg.inv(transforms["source_yup_to_world"]), world
        )
        np.testing.assert_allclose(recovered, points, atol=1e-8)
        identity = (
            transforms["world_to_camera_gravity"]
            @ transforms["camera_to_world_gravity"]
        )
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-8)

    def test_projection_uses_opencv_positive_depth(self):
        intrinsics = pinhole_intrinsics(512, np.deg2rad(60.0))
        points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, -1.0]])
        uv, depth, in_frame = project_world_points(
            points, np.eye(4), intrinsics, 512
        )
        np.testing.assert_allclose(uv[0], [256.0, 256.0])
        self.assertEqual(depth[0], 2.0)
        self.assertTrue(in_frame[0])
        self.assertFalse(in_frame[1])

    def test_skin_rotation_is_identity_at_rest_and_6d_uses_columns(self):
        half = np.deg2rad(35.0) / 2.0
        rest = np.asarray([[np.cos(half), 0.0, np.sin(half), 0.0]])
        matrix, six = skinning_rotations(rest[None], rest, np.eye(3))
        np.testing.assert_allclose(matrix[0, 0], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(six[0, 0], [1, 0, 0, 0, 1, 0], atol=1e-7)

        rotation = quaternion_wxyz_to_matrix(
            np.asarray([np.cos(half), 0, np.sin(half), 0])
        )
        expected = np.concatenate([rotation[:, 0], rotation[:, 1]])
        np.testing.assert_allclose(matrix_to_rotation_6d_columns(rotation), expected)
        matrix2, six2 = skinning_rotations_from_matrices(
            rotation[None, None], rotation[None], np.eye(3)
        )
        np.testing.assert_allclose(matrix2[0, 0], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(six2[0, 0], [1, 0, 0, 0, 1, 0], atol=1e-7)

    def test_evaluated_scale_and_shear_are_removed_from_rotation_targets(self):
        contaminated = np.asarray(
            [[1.001, 0.002, 0.0], [0.0, 0.999, -0.003], [0.0, 0.0, 1.0]]
        )
        projected = project_to_so3(contaminated)
        np.testing.assert_allclose(projected @ projected.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(projected), 1.0, places=12)
        matrix, _ = skinning_rotations_from_matrices(
            contaminated[None, None], contaminated[None], np.eye(3)
        )
        np.testing.assert_allclose(matrix[0, 0], np.eye(3), atol=1e-12)

    def test_action_name_recovers_gltf_split_source_animation(self):
        row = {
            "clip_index": "2",
            "clip_name": "Armature|cowboy1@walking|BaseLayer",
        }
        actions = ["Armature|cowboy1@walking|BaseLayer_Armature"]
        self.assertEqual(_select_action(row, actions), actions[0])

    def test_random_camera_yaw_is_deterministic_and_clip_specific(self):
        first = deterministic_camera_azimuth("clip-a", 20261010)
        repeated = deterministic_camera_azimuth("clip-a", 20261010)
        second = deterministic_camera_azimuth("clip-b", 20261010)
        self.assertEqual(first, repeated)
        self.assertGreaterEqual(first, -180.0)
        self.assertLess(first, 180.0)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
