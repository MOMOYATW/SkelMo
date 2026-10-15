"""Checks for the released checkpoint's image-to-joint alignment."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data_loaders.static_camera_dataset import _sample_dense_joint_features
from sample.generate_checkpoint import build_item


class DenseJointSamplingTest(unittest.TestCase):
    def test_bilinear_projection_and_out_of_frame_mask(self):
        patches = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        uv = np.asarray([[1.5, 1.5], [3.0, 3.0]])
        visible = np.asarray([True, False])
        sampled = _sample_dense_joint_features(patches, uv, visible, resolution=4)
        np.testing.assert_allclose(sampled[:, 0], [2.5, 0.0])

    def test_rejects_non_square_patch_grid(self):
        with self.assertRaises(ValueError):
            _sample_dense_joint_features(
                np.zeros((3, 128)), np.zeros((1, 2)), np.ones(1, bool), 512
            )

    def test_rest_features_keep_rest_bone_offsets(self):
        parents = np.asarray([-1, 0, 1])
        first_pose = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
        rest_pose = np.asarray([[10, 0, 0], [10, 1, 0], [10, 2, 0]], dtype=np.float32)
        values = {
            "rest_pose": rest_pose,
            "rest_appearance": np.ones((3, 768), dtype=np.float32),
        }
        stats = {"position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
        with TemporaryDirectory() as temporary:
            pca_path = Path(temporary) / "pca.npz"
            np.savez(pca_path, mean=np.zeros(768), components=np.eye(768, 128))
            item = build_item(
                np.zeros((1, 1369, 128), dtype=np.float16), parents,
                first_pose, np.zeros((3, 2)), np.ones(3, bool), 512, stats,
                "rest_merged", values, pca_path,
            )
        np.testing.assert_allclose(item[4], [[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(item[8].shape, (3, 256))
        np.testing.assert_allclose(np.linalg.norm(item[8][:, :128], axis=1), 1)


if __name__ == "__main__":
    unittest.main()
