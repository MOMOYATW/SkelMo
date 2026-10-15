"""CPU checks for the portable render/first-frame input preparation."""

import unittest
from types import SimpleNamespace

import numpy as np

from preprocess.static_camera_pilot.render_checkpoint_input import (
    fit_camera_distance,
    parent_first_bones,
)


class CheckpointRenderInputTest(unittest.TestCase):
    def test_helper_pruning_keeps_parent_first_tree(self):
        root = SimpleNamespace(name="root", parent=None)
        helper = SimpleNamespace(name="helper", parent=root)
        child = SimpleNamespace(name="child", parent=helper)
        leaf = SimpleNamespace(name="leaf", parent=child)
        armature = SimpleNamespace(data=SimpleNamespace(bones=[root, helper, child, leaf]))
        names, parents = parent_first_bones(armature, ["root", "child", "leaf"])
        self.assertEqual(names, ["root", "child", "leaf"])
        np.testing.assert_array_equal(parents, [-1, 0, 1])

    def test_invalid_joint_order_is_rejected(self):
        root = SimpleNamespace(name="root", parent=None)
        child = SimpleNamespace(name="child", parent=root)
        armature = SimpleNamespace(data=SimpleNamespace(bones=[root, child]))
        with self.assertRaisesRegex(ValueError, "parent-first"):
            parent_first_bones(armature, ["child", "root"])

    def test_training_camera_distance_fit(self):
        self.assertAlmostEqual(fit_camera_distance(1.5), 1.6484375)
        self.assertEqual(fit_camera_distance(2.0), 2.0)


if __name__ == "__main__":
    unittest.main()
