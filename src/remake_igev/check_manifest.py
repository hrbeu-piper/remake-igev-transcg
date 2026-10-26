"""Server-only synthetic checks for TransCG manifest boundaries.

Run with TMPDIR on the verified server resource disk. No real images are read.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .transcg_manifest import (CAMERAS, OFFICIAL_COUNTS, ManifestError,
                                  build_manifest, write_manifest)
except ImportError:
    from transcg_manifest import (CAMERAS, OFFICIAL_COUNTS, ManifestError,
                                 build_manifest, write_manifest)


def make_fixture(root: Path) -> None:
    """Official-size metadata only, with paired and unpaired viewpoints."""
    metadata = dict(OFFICIAL_COUNTS)
    metadata.update(train=list(range(1, 79)), test=list(range(79, 131)))
    types = {"isolated": [], "cluttered": []}
    for split in ("train", "test"):
        scene_ids = metadata[split]
        for index, scene_id in enumerate(scene_ids):
            kind = "isolated" if scene_id % 2 else "cluttered"
            types[kind].append(scene_id)
            scene = {"split": split, "type": kind}
            for camera, config in CAMERAS.items():
                total = metadata[f"{split}_{config['name']}_samples"]
                quotient, remainder = divmod(total, len(scene_ids))
                count = quotient + (index < remainder)
                # Offset camera 2 so both kinds of single-view fallback occur.
                start = 0 if camera == 1 else 240 - count
                scene[f"{config['name']}_valid_perspective_list"] = list(range(start, start + count))
                scene[f"{config['name']}_valid_perspective_num"] = count
            directory = root / f"scene{scene_id}"
            directory.mkdir(parents=True)
            (directory / "metadata.json").write_text(json.dumps(scene), encoding="utf-8")
    metadata.update(types)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class ManifestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="transcg-manifest-check-")
        cls.base = Path(cls.temp.name)
        cls.dataset = cls.base / "dataset"
        cls.dataset.mkdir()
        make_fixture(cls.dataset)
        cls.manifest = build_manifest(cls.dataset)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_scene_split_deterministic_and_disjoint(self) -> None:
        again = build_manifest(self.dataset)
        self.assertEqual(self.manifest, again)
        split = self.manifest["split_scenes"]
        self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [68, 10, 52])
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertEqual(split["test"], list(range(79, 131)))
        self.assertEqual(self.manifest["summary"]["val"]["scene_types"],
                         {"isolated": 5, "cluttered": 5})
        changed = build_manifest(self.dataset, seed=20261025)
        self.assertNotEqual(changed["split_scenes"]["val"], split["val"])
        self.assertEqual(changed["split_scenes"]["test"], split["test"])

    def test_reference_pairs_fallback_units_and_no_gt_inputs(self) -> None:
        rows = self.manifest["samples"]
        self.assertEqual(len(rows), 57715)
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(len(by_id), len(rows))
        self.assertTrue(any(row["mode"] == "single_view_fallback" for row in rows))
        for row in rows:
            camera = row["reference_camera_id"]
            self.assertEqual(row["inputs"]["raw_depth_scale_m"], {1: .001, 2: .00025}[camera])
            serialized_inputs = json.dumps(row["inputs"])
            self.assertNotIn("-gt", serialized_inputs)
            self.assertNotIn("supervision", serialized_inputs)
            source = row["inputs"]["source"]
            if row["mode"] == "paired":
                other_id = row["sample_id"][:-1] + str(3 - camera)
                other = by_id[other_id]
                self.assertEqual(other["split"], row["split"])
                self.assertEqual(other["mode"], "paired")
                self.assertEqual(source["reference_to_source_transform"]["invert_file_matrix"], camera == 2)
            else:
                self.assertIsNone(source)
        self.assertEqual(self.manifest["verification"]["level"], "metadata_only")
        self.assertFalse(self.manifest["verification"]["upload_completion_verified"])

    def _reject_mutated_metadata(self, relative: str, mutate) -> None:
        path = self.dataset / relative
        original = path.read_bytes()
        try:
            value = json.loads(original)
            mutate(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ManifestError):
                build_manifest(self.dataset)
        finally:
            path.write_bytes(original)

    def test_duplicate_view_ids_rejected(self) -> None:
        self._reject_mutated_metadata("scene1/metadata.json", lambda value:
                                     value["D435_valid_perspective_list"].append(0))

    def test_total_count_mismatch_rejected(self) -> None:
        self._reject_mutated_metadata("metadata.json", lambda value:
                                     value.update(train_samples=34190))
        def remove_one(value):
            value["D435_valid_perspective_list"].pop()
            value["D435_valid_perspective_num"] -= 1
        self._reject_mutated_metadata("scene1/metadata.json", remove_one)

    def test_scene_membership_mismatch_rejected(self) -> None:
        self._reject_mutated_metadata("scene1/metadata.json", lambda value:
                                     value.update(split="test"))
        self._reject_mutated_metadata("metadata.json", lambda value:
                                     value["test"].__setitem__(0, 1))

    def test_missing_and_malformed_metadata_rejected(self) -> None:
        path = self.dataset / "scene1/metadata.json"
        original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(ManifestError):
                build_manifest(self.dataset)
            path.write_text('{"type": "cluttered", "type": "isolated"}', encoding="utf-8")
            with self.assertRaises(ManifestError):
                build_manifest(self.dataset)
            path.write_text("not valid JSON", encoding="utf-8")
            with self.assertRaises(ManifestError):
                build_manifest(self.dataset)
        finally:
            path.write_bytes(original)

    def test_file_verification_is_explicit_and_rejects_missing_files(self) -> None:
        with self.assertRaisesRegex(ManifestError, "Missing or empty"):
            build_manifest(self.dataset, verify_files=True)

    def test_output_guard_idempotence_and_overwrite(self) -> None:
        output = self.base / "manifests/train.json"
        self.assertEqual(write_manifest(self.manifest, output, resource_root=self.base), "written")
        self.assertEqual(write_manifest(self.manifest, output, resource_root=self.base), "unchanged")
        changed = copy.copy(self.manifest)
        changed["seed"] += 1
        with self.assertRaisesRegex(ManifestError, "overwrite"):
            write_manifest(changed, output, resource_root=self.base)
        self.assertEqual(write_manifest(changed, output, resource_root=self.base, overwrite=True), "written")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["seed"], changed["seed"])
        with self.assertRaises(ManifestError):
            write_manifest(self.manifest, self.base.parent / "outside.json", resource_root=self.base)
        with self.assertRaises(ManifestError):
            write_manifest(self.manifest, self.dataset / "metadata.json", resource_root=self.base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
