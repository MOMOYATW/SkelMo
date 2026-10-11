from types import SimpleNamespace
import unittest

import numpy as np
import torch

from data_loaders.static_camera_dataset import (
    _conditioning_pose,
    _prepend_conditioning_video_token,
    _position_target,
    _root_aligned_rest,
    _structural_joint_features,
    topology_balancing_weights,
)
from data_loaders.tensors import truebones_batch_collate
from model.anytop import classifier_free_condition_mask
from utils.model_util import get_gmdm_args


class StaticCameraDatasetTests(unittest.TestCase):
    def test_classifier_free_mask_preserves_reference_frame(self):
        condition = torch.ones((32, 3, 4, 5))
        torch.manual_seed(7)
        masked = classifier_free_condition_mask(condition, 0.5, training=True)
        per_sample = masked.flatten(1).sum(1)
        self.assertTrue(torch.all((per_sample == 20) | (per_sample == 60)))
        self.assertTrue(torch.any(per_sample == 20))
        self.assertTrue(torch.any(per_sample == 60))
        self.assertIs(
            classifier_free_condition_mask(condition, 0.5, training=False),
            condition,
        )

    def test_full_topology_balancing_equalizes_bin_mass(self):
        records = [
            {"joint_count": 4},
            {"joint_count": 8},
            {"joint_count": 12},
            {"joint_count": 120},
        ]
        weights = topology_balancing_weights(records, power=1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]), places=6)

    def test_root_alignment_matches_first_pose_and_uniform_bone_scale(self):
        parents = np.asarray([-1, 0, 1])
        rest = np.asarray([[10, 0, 0], [10, 2, 0], [10, 4, 0]], dtype=np.float32)
        pose = np.asarray([[1, 2, 3], [1, 3, 3], [1, 4, 3]], dtype=np.float32)
        aligned, scale = _root_aligned_rest(rest, pose, parents)
        np.testing.assert_allclose(aligned[0], pose[0])
        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(aligned, axis=0), axis=-1), [1, 1]
        )

    def test_structural_features_do_not_require_a_rest_image(self):
        parents = np.asarray([-1, 0, 1])
        rest = np.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        features = _structural_joint_features(rest, parents)
        self.assertEqual(features.shape, (3, 768))
        self.assertTrue(np.isfinite(features).all())

    def test_first_frame_condition_uses_matching_numeric_and_visual_tokens(self):
        motion = np.arange(3 * 2 * 9, dtype=np.float32).reshape(3, 2, 9)
        rest = np.full((2, 9), -1, dtype=np.float32)
        video = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

        pose = _conditioning_pose(motion, rest, "first_frame")
        conditioned_video = _prepend_conditioning_video_token(video, "first_frame")

        np.testing.assert_array_equal(pose, motion[0])
        np.testing.assert_array_equal(conditioned_video[0], video[0])
        np.testing.assert_array_equal(conditioned_video[1], video[0])
        self.assertEqual(conditioned_video.shape, (4, 4, 5))

    def test_rest_condition_keeps_null_visual_token(self):
        motion = np.ones((3, 2, 9), dtype=np.float32)
        rest = np.full((2, 9), -1, dtype=np.float32)
        video = np.ones((3, 4, 5), dtype=np.float32)

        pose = _conditioning_pose(motion, rest, "rest")
        conditioned_video = _prepend_conditioning_video_token(video, "rest")

        np.testing.assert_array_equal(pose, rest)
        np.testing.assert_array_equal(conditioned_video[0], 0)
        np.testing.assert_array_equal(conditioned_video[1:], video)

    def test_first_frame_delta_target_is_zero_at_frame_zero(self):
        positions = np.asarray(
            [[[1, 2, 3], [4, 5, 6]], [[2, 4, 6], [3, 7, 8]]],
            dtype=np.float32,
        )
        std = np.asarray([2, 4, 5], dtype=np.float32)
        target = _position_target(positions, std, "first_frame_delta")
        np.testing.assert_array_equal(target[0], 0)
        np.testing.assert_allclose(target[1], [[0.5, 0.5, 0.6], [-0.5, 0.5, 0.4]])

    def test_position_rotation_model_uses_nine_channels(self):
        args = SimpleNamespace(
            latent_dim=128,
            ff_size=512,
            layers=2,
            heads=4,
            cond_mask_prob=0.1,
            value_emb=False,
            video_feature_mode="joint_tracks",
            video_track_dim=776,
            video_dense_dim=128,
            joint_feature_dim=256,
            motion_representation="positions_rot6d",
        )
        config = get_gmdm_args(args)
        self.assertEqual(config["feature_len"], 9)
        self.assertEqual(config["root_input_feats"], 9)

    def test_collate_preserves_normalization_statistics(self):
        frames, joints, features, max_joints = 2, 2, 9, 143
        mean = np.tile(np.arange(features, dtype=np.float32), (joints, 1))
        std = np.tile(np.arange(features, dtype=np.float32) + 1, (joints, 1))
        sample = (
            np.zeros((frames, joints, features), dtype=np.float32),
            frames,
            np.asarray([-1, 0], dtype=np.int64),
            np.zeros((joints, features), dtype=np.float32),
            np.zeros((joints, 3), dtype=np.float32),
            torch.ones((frames + 1, frames + 1), dtype=torch.bool),
            np.zeros((joints, joints), dtype=np.float32),
            np.zeros((joints, joints), dtype=np.float32),
            np.zeros((joints, 256), dtype=np.float32),
            0,
            mean,
            std,
            np.zeros((frames + 1, 1369, 128), dtype=np.float16),
            max_joints,
            "dense_dino",
        )
        _, condition = truebones_batch_collate([sample])
        np.testing.assert_allclose(condition["y"]["mean"][0, :joints], mean)
        np.testing.assert_allclose(condition["y"]["std"][0, :joints], std)
        np.testing.assert_allclose(condition["y"]["mean"][0, joints:], 0)
        np.testing.assert_allclose(condition["y"]["std"][0, joints:], 1)


if __name__ == "__main__":
    unittest.main()
