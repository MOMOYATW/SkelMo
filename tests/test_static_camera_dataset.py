from types import SimpleNamespace
import unittest

import numpy as np

from data_loaders.static_camera_dataset import (
    _root_aligned_rest,
    _structural_joint_features,
)
from utils.model_util import get_gmdm_args


class StaticCameraDatasetTests(unittest.TestCase):
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
            motion_representation="positions_rot6d",
        )
        config = get_gmdm_args(args)
        self.assertEqual(config["feature_len"], 9)
        self.assertEqual(config["root_input_feats"], 9)


if __name__ == "__main__":
    unittest.main()
