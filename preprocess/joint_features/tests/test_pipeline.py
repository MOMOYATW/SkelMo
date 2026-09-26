import json
import sys
import tempfile
import unittest
from pathlib import Path

from preprocess.joint_features.pipeline import parse_stages, run_command, validate_outputs
from preprocess.joint_features.views import DEFAULT_VIEW_NAMES, parse_view_names


class RestPoseDinoViewsTest(unittest.TestCase):
    def test_default_view_set_has_six_axes_and_eight_diagonals(self):
        self.assertEqual(len(DEFAULT_VIEW_NAMES), 14)
        self.assertEqual(parse_view_names(",".join(DEFAULT_VIEW_NAMES)), list(DEFAULT_VIEW_NAMES))

    def test_invalid_views_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_view_names("front,not_a_camera")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_view_names("front,front")

    def test_stages_are_canonicalized(self):
        self.assertEqual(parse_stages("extract,render"), ["render", "extract"])
        self.assertEqual(parse_stages("all"), ["render", "project", "extract", "validate"])

    def test_failed_stage_is_returned_for_reporting(self):
        result = run_command(
            "test", [sys.executable, "-c", "raise SystemExit(3)"], False
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 3)


class RestPoseDinoValidationTest(unittest.TestCase):
    def test_complete_animal_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render = root / "render"
            geometry = root / "geometry"
            features = root / "features"
            animal = "TestAnimal"
            render_dir = render / animal
            geometry_dir = geometry / animal
            feature_dir = features / animal
            for directory in (render_dir, geometry_dir, feature_dir):
                directory.mkdir(parents=True)
            metadata = {
                "views": list(DEFAULT_VIEW_NAMES),
                "view_cameras": {
                    view: {"matrix_world": [[1.0] * 4] * 4}
                    for view in DEFAULT_VIEW_NAMES
                },
            }
            (render_dir / "metadata.json").write_text(json.dumps(metadata))
            (render_dir / "rest_pose.glb").write_bytes(b"glb")
            for view in DEFAULT_VIEW_NAMES:
                (render_dir / f"{view}.png").write_bytes(b"png")
            (geometry_dir / "geometry.npz").write_bytes(b"npz")
            (geometry_dir / "geometry_metadata.json").write_text("{}")
            (feature_dir / "features.npz").write_bytes(b"npz")
            (feature_dir / "view_features.npz").write_bytes(b"npz")
            (feature_dir / "feature_metadata.json").write_text("{}")
            (feature_dir / "joint_averaged_features.npy").write_bytes(b"npy")
            (feature_dir / "joint_max_weight_features.npy").write_bytes(b"npy")
            result = validate_outputs(
                render,
                geometry,
                features,
                [animal],
                list(DEFAULT_VIEW_NAMES),
                True,
            )
            self.assertEqual(result["error_count"], 0)
            self.assertEqual(result["warning_count"], 0)
            self.assertEqual(result["ok"], 1)

    def test_legacy_camera_metadata_warns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            animal_dir = root / "render" / "TestAnimal"
            animal_dir.mkdir(parents=True)
            (animal_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "views": list(DEFAULT_VIEW_NAMES),
                        "camera_distance": 4.0,
                        "view_cameras": {},
                    }
                )
            )
            result = validate_outputs(
                root / "render",
                root / "geometry",
                root / "features",
                ["TestAnimal"],
                list(DEFAULT_VIEW_NAMES),
                True,
            )
            self.assertGreater(result["error_count"], 0)  # Other artifacts are absent.
            self.assertEqual(result["warning_count"], 1)
            self.assertTrue(
                any(
                    "legacy camera metadata" in warning
                    for warning in result["results"][0]["warnings"]
                )
            )

    def test_missing_all_camera_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            animal_dir = root / "render" / "TestAnimal"
            animal_dir.mkdir(parents=True)
            (animal_dir / "metadata.json").write_text(
                json.dumps({"views": list(DEFAULT_VIEW_NAMES), "view_cameras": {}})
            )
            result = validate_outputs(
                root / "render",
                root / "geometry",
                root / "features",
                ["TestAnimal"],
                list(DEFAULT_VIEW_NAMES),
                True,
            )
            self.assertTrue(
                any(
                    "neither per-view matrix_world" in error
                    for error in result["results"][0]["errors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
