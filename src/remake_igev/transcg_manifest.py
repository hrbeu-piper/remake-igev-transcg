"""Build deterministic TransCG scene splits without modifying source metadata.

Run on a project server. Metadata-only verification does not establish that an
upload is complete. File verification checks existence and nonzero size, not
image decoding or checksums; the transfer's checksum receipt remains required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any


DEFAULT_SEED = 20261024
VALIDATION_SCENES = 10
CAMERAS = {
    1: {"name": "D435", "depth_scale_m": 0.001,
        "intrinsics": "camera_intrinsics/1-camIntrinsics-D435.npy"},
    2: {"name": "L515", "depth_scale_m": 0.00025,
        "intrinsics": "camera_intrinsics/2-camIntrinsics-L515.npy"},
}
OFFICIAL_COUNTS = {
    "total_scenes": 130, "perspective_num": 240, "total_samples": 57715,
    "total_D435_samples": 28967, "total_L515_samples": 28748,
    "train_samples": 34191, "train_D435_samples": 17201,
    "train_L515_samples": 16990, "test_samples": 23524,
    "test_D435_samples": 11766, "test_L515_samples": 11758,
}
EXTRINSICS = "T_camera2_camera1.npy"


class ManifestError(ValueError):
    """Malformed data or unsafe manifest destination."""


def _object_without_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ManifestError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_metadata(path: Path, root: Path, receipts: dict[str, str]) -> dict:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"),
                           object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError(f"Cannot read valid metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Expected JSON object: {path}")
    receipts[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _ids(value: Any, label: str, low: int, high: int) -> list[int]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a list")
    values = [_integer(item, label, low) for item in value]
    if len(set(values)) != len(values):
        raise ManifestError(f"{label} contains duplicate IDs")
    if any(item >= high for item in values):
        raise ManifestError(f"{label} has an ID outside [{low}, {high})")
    return sorted(values)


def _validation_ids(train_ids: list[int], scenes: dict[int, dict], seed: int) -> list[int]:
    """Proportional allocation by scene type, then seeded selection within type."""
    groups: dict[str, list[int]] = {}
    for scene_id in train_ids:
        groups.setdefault(scenes[scene_id]["type"], []).append(scene_id)
    quotas = {kind: len(ids) * VALIDATION_SCENES // len(train_ids)
              for kind, ids in groups.items()}
    remaining = VALIDATION_SCENES - sum(quotas.values())
    priority = sorted(groups, key=lambda kind: (
        -(len(groups[kind]) * VALIDATION_SCENES % len(train_ids)), kind))
    for kind in priority[:remaining]:
        quotas[kind] += 1
    rng = random.Random(seed)
    selected = []
    for kind in sorted(groups):
        selected.extend(rng.sample(sorted(groups[kind]), quotas[kind]))
    return sorted(selected)


def _sample(split: str, scene_id: int, view_id: int,
            camera: int, paired: bool, scene_type: str) -> dict:
    directory = f"scene{scene_id}/{view_id}"
    config = CAMERAS[camera]
    source_camera = 3 - camera
    inputs: dict[str, Any] = {
        "rgb": f"{directory}/rgb{camera}.png",
        "raw_depth": f"{directory}/depth{camera}.png",
        "intrinsics": config["intrinsics"],
        "raw_depth_scale_m": config["depth_scale_m"],
        "source": None,
    }
    if paired:
        inputs["source"] = {
            "camera_id": source_camera,
            "rgb": f"{directory}/rgb{source_camera}.png",
            "intrinsics": CAMERAS[source_camera]["intrinsics"],
            "reference_to_source_transform": {
                "path": EXTRINSICS,
                "invert_file_matrix": camera == 2,
                "direction_status": "provisional_pending_geometry_gate",
            },
        }
    return {
        "sample_id": f"scene{scene_id:03d}/view{view_id:03d}/camera{camera}",
        "split": split, "scene_id": scene_id, "view_id": view_id,
        "scene_type": scene_type, "reference_camera_id": camera,
        "mode": "paired" if paired else "single_view_fallback",
        "inputs": inputs,
        # Ground truth is deliberately separate from forward inputs. Choosing a
        # privileged mask-input experiment requires an explicit later protocol.
        "supervision": {
            "depth_gt": f"{directory}/depth{camera}-gt.png",
            "transparent_mask": f"{directory}/depth{camera}-gt-mask.png",
            "depth_gt_scale_m": config["depth_scale_m"],
        },
    }


def _file_paths(row: dict) -> set[str]:
    inputs, supervision = row["inputs"], row["supervision"]
    paths = {inputs["rgb"], inputs["raw_depth"], inputs["intrinsics"],
             supervision["depth_gt"], supervision["transparent_mask"]}
    if inputs["source"] is not None:
        source = inputs["source"]
        paths.update((source["rgb"], source["intrinsics"],
                      source["reference_to_source_transform"]["path"]))
    return paths


def build_manifest(dataset_root: str | Path, *, seed: int = DEFAULT_SEED,
                   verify_files: bool = False) -> dict:
    """Validate official metadata and return 68/10/52 scene manifests in memory.

    The official test scene IDs are preserved exactly. No calibration is fitted,
    and no images are read unless ``verify_files`` is explicitly requested.
    """
    _integer(seed, "seed")
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ManifestError(f"Dataset root is not a directory: {root}")
    receipts: dict[str, str] = {}
    metadata = _read_metadata(root / "metadata.json", root, receipts)
    for key, expected in OFFICIAL_COUNTS.items():
        actual = _integer(metadata.get(key), key)
        if actual != expected:
            raise ManifestError(f"Official {key}: expected {expected}, found {actual}")
    train_ids = _ids(metadata.get("train"), "train scenes", 1, 131)
    test_ids = _ids(metadata.get("test"), "test scenes", 1, 131)
    if len(train_ids) != 78 or len(test_ids) != 52:
        raise ManifestError("Expected official 78 train and 52 test scenes")
    if set(train_ids) & set(test_ids):
        raise ManifestError("Original train and test scene IDs overlap")
    if set(train_ids) | set(test_ids) != set(range(1, 131)):
        raise ManifestError("Original splits must cover exactly scenes 1..130")

    scenes = {}
    counts = {split: {camera: 0 for camera in CAMERAS} for split in ("train", "test")}
    for scene_id in range(1, 131):
        scene = _read_metadata(root / f"scene{scene_id}/metadata.json", root, receipts)
        expected_split = "train" if scene_id in train_ids else "test"
        if scene.get("split") != expected_split:
            raise ManifestError(f"scene{scene_id} split disagrees with root metadata")
        if scene.get("type") not in ("isolated", "cluttered"):
            raise ManifestError(f"scene{scene_id} has unsupported scene type")
        scene["views"] = {}
        for camera, camera_info in CAMERAS.items():
            prefix = camera_info["name"]
            key = f"{prefix}_valid_perspective_list"
            ids = _ids(scene.get(key), f"scene{scene_id}/{key}", 0, 240)
            declared = _integer(scene.get(f"{prefix}_valid_perspective_num"), key)
            if declared != len(ids):
                raise ManifestError(f"scene{scene_id}/{prefix} perspective count mismatch")
            scene["views"][camera] = set(ids)
            counts[expected_split][camera] += len(ids)
        scenes[scene_id] = scene
    for split, cameras in counts.items():
        for camera, count in cameras.items():
            key = f"{split}_{CAMERAS[camera]['name']}_samples"
            if count != metadata[key]:
                raise ManifestError(f"{key}: scene totals {count} != metadata {metadata[key]}")
        if sum(cameras.values()) != metadata[f"{split}_samples"]:
            raise ManifestError(f"{split}: sample total mismatch")
    for kind in ("isolated", "cluttered"):
        if kind in metadata:
            declared = _ids(metadata[kind], f"{kind} scenes", 1, 131)
            actual = sorted(s for s, scene in scenes.items() if scene["type"] == kind)
            if declared != actual:
                raise ManifestError(f"{kind}: root and scene type assignments disagree")

    val_ids = _validation_ids(train_ids, scenes, seed)
    split_scenes = {"train": sorted(set(train_ids) - set(val_ids)),
                    "val": val_ids, "test": test_ids}
    rows: list[dict] = []
    split_summary = {}
    for split, scene_ids in split_scenes.items():
        split_rows = []
        types = {kind: 0 for kind in ("isolated", "cluttered")}
        for scene_id in scene_ids:
            scene = scenes[scene_id]
            types[scene["type"]] += 1
            a, b = scene["views"][1], scene["views"][2]
            for view in sorted(a | b):
                for camera in CAMERAS:
                    if view in scene["views"][camera]:
                        split_rows.append(_sample(split, scene_id, view, camera,
                                                  view in a and view in b, scene["type"]))
        paired = sum(row["mode"] == "paired" for row in split_rows)
        split_summary[split] = {
            "scene_count": len(scene_ids), "scene_types": types,
            "samples": len(split_rows), "paired_reference_samples": paired,
            "paired_viewpoints": paired // 2,
            "single_view_fallback_samples": len(split_rows) - paired,
            "camera_samples": {str(cam): sum(row["reference_camera_id"] == cam
                                              for row in split_rows) for cam in CAMERAS},
        }
        rows.extend(split_rows)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ManifestError("Duplicate sample IDs after manifest construction")
    if len(rows) != metadata["total_samples"]:
        raise ManifestError("Manifest sample total disagrees with official metadata")
    if len(split_scenes["train"]) != 68 or len(split_scenes["val"]) != 10:
        raise ManifestError("Scene allocation failed the 68/10 training/validation split")
    if any(set(split_scenes[a]) & set(split_scenes[b])
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ManifestError("Generated scene splits overlap")

    checked_files = 0
    if verify_files:
        required = {path for row in rows for path in _file_paths(row)}
        for relative in sorted(required):
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ManifestError(f"Missing or empty required dataset file: {path}")
        checked_files = len(required)
    receipt_text = json.dumps(receipts, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1, "dataset": "TransCG", "dataset_root": str(root),
        "seed": seed, "split_method": "scene_type_stratified_largest_remainder",
        "validation_scene_count": VALIDATION_SCENES,
        "original_split_scenes": {"train": train_ids, "test": test_ids},
        "split_scenes": split_scenes, "summary": split_summary,
        "camera_depth_scales_m": {str(c): info["depth_scale_m"] for c, info in CAMERAS.items()},
        "calibration_status": "provisional_pending_geometry_gate",
        "mask_input_policy": "no_ground_truth_mask_in_inputs",
        "verification": {
            "level": "metadata_and_file_existence" if verify_files else "metadata_only",
            "metadata_files": len(receipts), "required_files_checked": checked_files,
            "content_checksums_verified": False, "upload_completion_verified": False,
            "metadata_sha256": receipts,
            "metadata_snapshot_sha256": hashlib.sha256(receipt_text.encode("utf-8")).hexdigest(),
        },
        "samples": rows,
    }


def write_manifest(manifest: dict, output: str | Path, *,
                   resource_root: str | Path, overwrite: bool = False) -> str:
    """Write atomically below an explicitly supplied, already existing resource root.

    Mount identity, writable space, and completed transfer must be checked by the
    server launcher. This function verifies path containment, not disk identity.
    """
    base = Path(resource_root).expanduser()
    destination = Path(output).expanduser()
    if not base.is_absolute() or not destination.is_absolute():
        raise ManifestError("Output and resource root must be absolute server paths")
    base = base.resolve(strict=True)
    if not base.is_dir() or base == Path(base.anchor):
        raise ManifestError("Resource root must be an existing non-root directory")
    if destination.is_symlink():
        raise ManifestError("Refusing a manifest output symlink")
    destination = destination.resolve(strict=False)
    if not destination.is_relative_to(base) or destination == base:
        raise ManifestError("Manifest output must be strictly inside resource root")
    dataset_root = Path(manifest["dataset_root"]).resolve(strict=True)
    if destination.is_relative_to(dataset_root):
        raise ManifestError("Manifest output must be outside the raw dataset directory")
    payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if destination.exists():
        if not destination.is_file():
            raise ManifestError("Manifest output already exists and is not a file")
        if destination.read_bytes() == payload:
            return "unchanged"
        if not overwrite:
            raise ManifestError("Different output exists; explicit --overwrite is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".transcg-manifest-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.is_file() and destination.read_bytes() == payload:
                    return "unchanged"
                raise ManifestError("Output appeared concurrently; refusing overwrite")
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--output", help="Absolute manifest path; omit for a read-only stdout summary")
    parser.add_argument("--resource-root", help="Existing server resource root; required with --output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output and not args.resource_root:
        parser.error("--output requires an explicit --resource-root")
    if not args.output and (args.resource_root or args.overwrite):
        parser.error("--resource-root and --overwrite require --output")
    try:
        manifest = build_manifest(args.dataset_root, seed=args.seed, verify_files=args.verify_files)
        status = "not_written"
        if args.output:
            status = write_manifest(manifest, args.output, resource_root=args.resource_root,
                                    overwrite=args.overwrite)
    except (ManifestError, OSError) as exc:
        parser.exit(2, f"Manifest error: {exc}\n")
    print(json.dumps({
        "dataset_root": manifest["dataset_root"], "seed": manifest["seed"],
        "split_scenes": manifest["split_scenes"], "summary": manifest["summary"],
        "verification": {key: value for key, value in manifest["verification"].items()
                         if key != "metadata_sha256"},
        "output": args.output, "write_status": status,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
