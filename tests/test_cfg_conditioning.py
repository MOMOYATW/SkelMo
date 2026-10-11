import unittest

import torch

from model.anytop import classifier_free_condition_mask


class ClassifierFreeConditionMaskTest(unittest.TestCase):
    def setUp(self):
        self.video = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)

    def test_forced_unconditional_keeps_first_frame(self):
        masked = classifier_free_condition_mask(
            self.video,
            probability=0.0,
            training=True,
            preserve_prefix_tokens=1,
            force_drop=torch.tensor([True, False]),
        )
        torch.testing.assert_close(masked[0, 0], self.video[0, 0])
        self.assertEqual(masked[0, 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(masked[1], self.video[1])

    def test_probability_one_drops_only_motion_frames(self):
        masked = classifier_free_condition_mask(
            self.video,
            probability=1.0,
            training=True,
            preserve_prefix_tokens=1,
        )
        torch.testing.assert_close(masked[:, 0], self.video[:, 0])
        self.assertEqual(masked[:, 1:].count_nonzero().item(), 0)

    def test_eval_does_not_randomly_mask(self):
        masked = classifier_free_condition_mask(
            self.video,
            probability=1.0,
            training=False,
            preserve_prefix_tokens=1,
        )
        self.assertIs(masked, self.video)


if __name__ == "__main__":
    unittest.main()
