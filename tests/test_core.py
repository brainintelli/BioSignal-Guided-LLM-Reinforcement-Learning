import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from src.access import archive_code
from src.exploration import exploration, load_pairs, paired_summary
from src.files import ROOT, Release, read_json, root_path, sha256


class PathTests(unittest.TestCase):
    def test_relative_path_stays_inside_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            root.mkdir()
            (root / "data").mkdir()
            self.assertEqual(root_path(root, "data/../table.csv"), root.resolve() / "table.csv")

    def test_escape_and_absolute_paths_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            root.mkdir()
            for name in ("../release-other/table.csv", str(root.resolve() / "table.csv")):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "relative to the release directory"):
                        root_path(root, name)


class PairLoadingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"seed_index": str(seed), "map_index": str(map_index)}
            for seed in (1, 2)
            for map_index in (1, 2)
        ]
        self.release = SimpleNamespace(spec={"short": {"seeds": 2, "maps": 2}}, rows=self.rows_for)

    def rows_for(self, key):
        self.assertEqual(key, "exploration")
        return self.rows

    def test_duplicate_with_complete_grid_is_rejected(self):
        self.rows.append(self.rows[0].copy())
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            load_pairs(self.release)

    def test_missing_seed_map_combination_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_pairs(self.release)


class PairedSummaryTests(unittest.TestCase):
    def test_all_ties(self):
        summary = paired_summary([2, 4, 6], [2, 4, 6])
        self.assertEqual((summary["wins"], summary["ties"], summary["losses"]), (0, 3, 0))
        self.assertEqual(summary["mean_difference"], 0)
        self.assertEqual(summary["percent_of_control"], 0)
        self.assertIsNone(summary["paired_dz"])
        self.assertIsNone(summary["rank_biserial"])

    def test_zero_baseline_and_constant_nonzero_difference(self):
        summary = paired_summary([0, 0, 0], [2, 2, 2])
        self.assertEqual(summary["mean_difference"], 2)
        self.assertIsNone(summary["percent_of_control"])
        self.assertIsNone(summary["paired_dz"])
        self.assertEqual(summary["rank_biserial"], 1)

    def test_signed_ranks_average_tied_magnitudes_and_skip_zero(self):
        summary = paired_summary([5] * 6, [5, 6, 4, 8, 8, 0])
        self.assertEqual((summary["wins"], summary["ties"], summary["losses"]), (3, 1, 2))
        # Average ranks are 1.5 for magnitude 1, 3.5 for magnitude 3, and 5 for magnitude 5.
        positive_rank_sum = 1.5 + 3.5 + 3.5
        negative_rank_sum = 1.5 + 5
        self.assertAlmostEqual(
            summary["rank_biserial"],
            (positive_rank_sum - negative_rank_sum) / (positive_rank_sum + negative_rank_sum),
        )

    def test_negative_observations_are_rejected_in_either_arm(self):
        for control, arm in (([-1, 2], [0, 2]), ([0, 2], [-1, 2])):
            with self.subTest(control=control, arm=arm):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    paired_summary(control, arm)

    def test_published_smoke_outcomes(self):
        expected = read_json(ROOT / "results" / "exploration.json")["d6"]["intact"]
        actual = exploration(Release())["d6"]["intact"]
        for field in ("pairs", "wins", "ties", "losses"):
            with self.subTest(field=field):
                self.assertEqual(actual[field], expected[field])
        self.assertAlmostEqual(actual["mean_difference"], expected["mean_difference"], places=11)


class InputIntegrityTests(unittest.TestCase):
    def test_check_inputs_rejects_corrupted_pin(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            config = root / "study.yaml"
            config.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            source = root / "data" / "toy.csv"
            source.write_bytes(b"count\n1\n")
            pins = {"data/toy.csv": sha256(source)}
            (root / "data" / "input_hashes.json").write_text(json.dumps(pins), encoding="utf-8")
            release = Release(root=root, config=config.name)
            release.check_inputs()

            source.write_bytes(b"count\n2\n")
            with self.assertRaisesRegex(ValueError, r"Input missing or changed: data/toy\.csv"):
                release.check_inputs()

    def test_archive_rejects_a_same_size_zip_with_a_different_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected.zip"
            supplied = root / "supplied.zip"
            member = "model.py"
            for path, body in ((expected, "value = 1\n"), (supplied, "value = 2\n")):
                with ZipFile(path, "w") as archive:
                    archive.writestr(member, body)
            self.assertEqual(expected.stat().st_size, supplied.stat().st_size)
            release = SimpleNamespace(
                spec={
                    "archived_model_code": {
                        "bytes": expected.stat().st_size,
                        "sha256": sha256(expected),
                    }
                }
            )
            destination = root / "output"
            destination.mkdir()

            with self.assertRaisesRegex(ValueError, "size and SHA-256"):
                archive_code(release, supplied, destination)
            self.assertEqual(list(destination.iterdir()), [])

            public_body = b"value = 2\n"
            release.spec["public_receipt_archive"] = {
                "bytes": supplied.stat().st_size,
                "sha256": sha256(supplied),
                "members": {member: hashlib.sha256(public_body).hexdigest()},
            }
            report = archive_code(release, supplied, destination)
            self.assertEqual(report["archive_sha256"], sha256(supplied))
            self.assertEqual((destination / "archived-code" / member).read_bytes(), public_body)


if __name__ == "__main__":
    unittest.main()
