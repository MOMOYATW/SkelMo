import unittest

import numpy as np
import torch

from data_loaders.truebones.data.dataset import (
    permute_non_root_joints,
    prune_leaf_joints,
    subdivide_joint_edge,
)
from preprocess.video_joint_tracks import match_dino_patches


class VideoJointTrackTests(unittest.TestCase):
    def test_match_shape_and_finite_statistics(self):
        patches = torch.randn(3, 8, 4, 4)
        queries = torch.randn(5, 8)
        support = torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
        tracks = match_dino_patches(patches, queries, support)
        self.assertEqual(tuple(tracks.shape), (3, 5, 16))
        self.assertTrue(torch.isfinite(tracks).all())
        torch.testing.assert_close(tracks[..., -1], support.expand(3, -1))

    def test_joint_permutation_updates_parent_indices(self):
        motion = np.arange(2 * 4 * 3).reshape(2, 4, 3)
        features = np.arange(4 * 2).reshape(4, 2)
        parents = np.asarray([-1, 0, 1, 1])
        relations = np.arange(16).reshape(4, 4)
        graph = relations + 100
        rng = np.random.RandomState(4)
        values, new_parents, new_relations, new_graph = permute_non_root_joints(
            [(motion, 1), (features, 0)], parents, relations, graph, rng=rng
        )
        new_motion, new_features = values
        self.assertTrue(np.array_equal(new_motion[:, 0], motion[:, 0]))
        self.assertTrue(np.array_equal(new_features[0], features[0]))
        self.assertEqual(new_parents[0], -1)
        for new_joint in range(1, 4):
            old_joint = np.flatnonzero((features == new_features[new_joint]).all(1))[0]
            old_parent = parents[old_joint]
            new_parent_old_slot = np.flatnonzero(
                (features == new_features[new_parents[new_joint]]).all(1)
            )[0]
            self.assertEqual(new_parent_old_slot, old_parent)
        self.assertEqual(new_relations.shape, relations.shape)
        self.assertEqual(new_graph.shape, graph.shape)

    def test_leaf_pruning_preserves_valid_tree(self):
        parents = np.asarray([-1, 0, 1, 1, 0])
        motion = np.zeros((3, 5, 3), dtype=np.float32)
        features = np.arange(10).reshape(5, 2)
        values, new_parents, relations, graph = prune_leaf_joints(
            [(motion, 1), (features, 0)],
            parents,
            rng=np.random.RandomState(2),
            fraction=0.5,
        )
        self.assertLess(len(new_parents), len(parents))
        self.assertEqual(new_parents[0], -1)
        self.assertTrue(np.all(new_parents[1:] >= 0))
        self.assertTrue(np.all(new_parents[1:] < len(new_parents)))
        self.assertEqual(values[0].shape[1], len(new_parents))
        self.assertEqual(relations.shape, (len(new_parents), len(new_parents)))
        self.assertEqual(graph.shape, relations.shape)

    def test_subdivision_preserves_old_endpoint_positions(self):
        parents = np.asarray([-1, 0, 1])
        motion = np.random.RandomState(1).randn(4, 3, 3).astype(np.float32)
        tpose = motion[0].copy()
        offsets = np.ones((3, 3), dtype=np.float32)
        features = np.ones((3, 8), dtype=np.float32)
        tracks = np.ones((4, 3, 16), dtype=np.float32)
        result = subdivide_joint_edge(
            motion, tpose, offsets, features, tracks, parents,
            rng=np.random.RandomState(3),
        )
        new_motion, _, new_offsets, _, new_tracks, new_parents, relations, graph = result
        np.testing.assert_allclose(new_motion[:, :3], motion)
        self.assertEqual(new_motion.shape[1], 4)
        self.assertEqual(new_offsets.shape[0], 4)
        self.assertEqual(new_tracks.shape[1], 4)
        self.assertEqual(len(new_parents), 4)
        self.assertEqual(relations.shape, (4, 4))
        self.assertEqual(graph.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
