import unittest
import tempfile
from pathlib import Path

import numpy as np

from preprocess.joint_features.extract_features import (
    fill_unsupported_joints,
    pool_dominant_joint_features,
    pool_joint_features,
    save_skelmo_joint_features,
)


class JointPoolingTest(unittest.TestCase):
    def test_skin_weighted_mean(self):
        features = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        indices = np.asarray([[0, 1], [0, -1], [1, -1]], dtype=np.int32)
        weights = np.asarray([[0.75, 0.25], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pooled, support = pool_joint_features(features, indices, weights, joint_count=3)
        np.testing.assert_allclose(support, [1.75, 1.25, 0.0])
        np.testing.assert_allclose(pooled[0], [0.75 / 1.75, 2.0 / 1.75])
        np.testing.assert_allclose(pooled[1], [3.25 / 1.25, 3.0 / 1.25])
        np.testing.assert_allclose(pooled[2], [0.0, 0.0])

    def test_fill_unsupported_joint(self):
        features = np.asarray([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]], dtype=np.float32)
        support = np.asarray([1.0, 0.0, 2.0], dtype=np.float32)
        positions = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
        filled, sources = fill_unsupported_joints(features, support, positions)
        np.testing.assert_array_equal(sources, [0, 0, 2])
        np.testing.assert_allclose(filled[1], features[0])

    def test_dominant_skin_influence_pooling(self):
        features = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        indices = np.asarray([[0, 1], [0, 1], [1, -1]], dtype=np.int32)
        weights = np.asarray([[0.75, 0.25], [0.4, 0.6], [1.0, 0.0]], dtype=np.float32)
        pooled, counts = pool_dominant_joint_features(features, indices, weights, 3)
        np.testing.assert_array_equal(counts, [1, 2, 0])
        np.testing.assert_allclose(pooled[0], [1.0, 0.0])
        np.testing.assert_allclose(pooled[1], [1.5, 2.5])
        np.testing.assert_allclose(pooled[2], [0.0, 0.0])

    def test_skelmo_dictionary_outputs(self):
        names = np.asarray(["root", "tail"])
        averaged = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dominant = np.asarray([[5.0, 6.0], [0.0, 0.0]], dtype=np.float32)
        counts = np.asarray([2, 0], dtype=np.int32)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            save_skelmo_joint_features(output_dir, names, averaged, dominant, counts)
            averaged_dict = np.load(
                output_dir / "joint_averaged_features.npy", allow_pickle=True
            ).item()
            dominant_dict = np.load(
                output_dir / "joint_max_weight_features.npy", allow_pickle=True
            ).item()
        self.assertEqual(set(averaged_dict), {"root", "tail"})
        self.assertEqual(set(dominant_dict), {"root"})
        np.testing.assert_allclose(averaged_dict["tail"], [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
