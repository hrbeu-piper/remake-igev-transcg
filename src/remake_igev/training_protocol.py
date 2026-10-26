"""Fail-closed provenance/configuration gates for formal TransCG B/G/F training.

This module never loads pickle/checkpoint tensors and never edits raw data.
Run validate_training_config(..., guard=guard), validate_gate_bundle(...) and
validate_parent(...) before model construction. A schema-only config check
without guard is not a launch authorization. CUDA execution is a separate gate.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .objectives import LossConfig
from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash, sha256_file
from .train_engine import LOSS_NORMALIZATION, SAMPLER_VERSION
from .transcg_manifest import _file_paths, _sample, build_manifest


OFFICIAL_TRAIN_SCENES = [1, 2, 3, 4, 6, 9, 10, 14, 17, 18, 19, 22, 25, 26, 28, 30, 31, 33, 34, 35,
                         37, 38, 39, 43, 44, 46, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 65, 67, 68,
                         69, 70, 71, 74, 75, 77, 80, 85, 89, 90, 92, 93, 100, 102, 103, 104, 105, 106, 107,
                         108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128]
VALIDATION_SCENES = [3, 9, 17, 38, 43, 58, 68, 77, 114, 122]
SPLITS = {"train": sorted(set(OFFICIAL_TRAIN_SCENES) - set(VALIDATION_SCENES)),
          "val": VALIDATION_SCENES, "test": sorted(set(range(1, 131)) - set(OFFICIAL_TRAIN_SCENES))}
COUNTS = {"train": (29861, 29202, 15005, 14856), "val": (4330, 4248, 2196, 2134),
          "test": (23524, 23006, 11766, 11758)}
BUDGETS = {"baseline": 40, "geometry": 5, "fusion": 15}
PARENT_STAGE = {"geometry": "baseline", "fusion": "geometry"}
DEFAULT_GEOMETRY = {"feature_channels": 64, "context_channels": [48, 96, 192],
                    "hidden_channels": 128, "volume_channels": 8, "geometry_channels": 32,
                    "lookup_radius": 2, "chunk_size": 8, "q_min": .5, "q_max": 10., "max_update": 1.}
METADATA_SNAPSHOT_SHA256 = "040b49cade2f73e076ce7efd6d530af114edb95719d5eddab78b599f667bd9b0"
CALIBRATION_FILES = ("camera_intrinsics/1-camIntrinsics-D435.npy",
                     "camera_intrinsics/2-camIntrinsics-L515.npy", "T_camera2_camera1.npy")
TRANSFER_SOURCE = "/media/bzl/系统/transcg"
# A complete receipt from the retired instance cannot authorize the replacement.
TRANSFER_HOST = "connect.nmb2.seetacloud.com"
TRANSFER_PORT = 28143


class ProtocolError(RuntimeSafetyError):
    """Evidence is missing, mismatched, diagnostic-only, or outside this protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _sha(value, label: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def _json(path: Path) -> dict:
    def pairs(items):
        output = {}
        for key, value in items:
            _require(key not in output, f"duplicate JSON key: {key}")
            output[key] = value
        return output
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError(f"nonfinite JSON: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read evidence {path}: {exc}") from exc
    _require(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def _inside_file(guard: ResourceGuard, value, label: str) -> Path:
    _require(isinstance(value, (str, Path)), f"missing {label} path")
    path = guard.path(value)
    _require(path.is_file(), f"missing {label} file: {path}")
    return path


def validate_training_config(config: Mapping, *, guard: ResourceGuard | None = None,
                             require_formal: bool = True) -> dict:
    """Return compatible config/hash plus execution controls; never weaken hashes.

    Only execution (physical batching/workers) and run_id are excluded from the
    compatible hash. Effective batch, sampler, normalization, loss, seed, phase
    budget, source hashes and initialization remain bound. OOM rebatching may
    change numerical RNG consumption; it is not a promise of bitwise replay.
    """
    _require(isinstance(config, Mapping), "training config must be an object")
    try:
        cfg = json.loads(json.dumps(dict(config), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"config must contain finite JSON values: {exc}") from exc
    _require(cfg.get("schema_version") == 1, "unsupported training config schema")
    mode = cfg.get("mode")
    _require(mode == "formal" if require_formal else mode in {"formal", "diagnostic"}, "diagnostic/reference config is not formal")
    stage = cfg.get("stage")
    _require(stage in BUDGETS, "stage must be baseline, geometry or fusion")
    _require(isinstance(cfg.get("run_id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", cfg["run_id"]) is not None,
             "invalid run_id")
    _integer(cfg.get("seed"), "seed")
    _integer(cfg.get("epochs"), "epochs", 1)
    if mode == "formal":
        _require(cfg["epochs"] == BUDGETS[stage], "formal B/G/F budgets are exactly 40/5/15 epochs")
    for key, expected in (("effective_batch", 8), ("output_hw", [480, 640]), ("precision", "fp32"),
                          ("mask_protocol", "gt"), ("augmentation", "identity"), ("iterations", 8),
                          ("sampler_version", SAMPLER_VERSION), ("loss_normalization", LOSS_NORMALIZATION)):
        _require(cfg.get(key) == expected, f"protocol mismatch: {key}")
    _integer(cfg.get("checkpoint_every_steps"), "checkpoint_every_steps", 1)
    _require(cfg.get("selection") == {"split": "val", "metric": "transparent_mae_mm", "direction": "min"},
             "checkpoint selection must use validation transparent pixel-weighted MAE")
    execution = cfg.get("execution")
    _require(isinstance(execution, dict), "execution controls required")
    physical = _integer(execution.get("physical_batch"), "physical_batch", 1)
    accumulation = _integer(execution.get("accumulation_steps"), "accumulation_steps", 1)
    _require(physical * accumulation == cfg["effective_batch"], "physical/accumulated batch does not preserve effective batch")
    _require(execution.get("workers") == 0, "protocol v1 requires synchronous workers=0")
    _require(set(execution) <= {"physical_batch", "accumulation_steps", "workers"}, "unknown execution control could hide semantic changes")
    for key in ("manifest_sha256", "gate_fingerprint"):
        _sha(cfg.get(key), key)
    geometry = cfg.get("geometry")
    _require(isinstance(geometry, dict) and set(geometry) == set(DEFAULT_GEOMETRY),
             "serialize all geometry options explicitly; implicit model defaults are not a compatible protocol")
    _require(geometry["feature_channels"] == 64 and geometry["context_channels"] == [48, 96, 192], "geometry/model channel wiring differs")
    for name in ("hidden_channels", "volume_channels", "geometry_channels"):
        _integer(geometry[name], f"geometry {name}", 2)
    _integer(geometry["lookup_radius"], "lookup_radius")
    _integer(geometry["chunk_size"], "chunk_size", 1)
    _require(all(type(geometry[name]) in (int, float) and math.isfinite(geometry[name])
                 for name in ("q_min", "q_max", "max_update"))
             and 0 < geometry["q_min"] < geometry["q_max"] and geometry["max_update"] > 0,
             "invalid geometry inverse-depth/update bounds")
    loss = cfg.get("loss")
    _require(isinstance(loss, dict) and set(loss) == set(asdict(LossConfig())), "serialize every LossConfig field explicitly")
    try:
        LossConfig(**loss)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"invalid loss config: {exc}") from exc
    _require(loss["geometry_q_min"] == geometry["q_min"] and loss["geometry_q_max"] == geometry["q_max"],
             "loss masks and model candidate ranges must agree")
    if stage == "baseline":
        _require(all(loss[name] == 0 for name in ("gate_weight", "residual_weight", "sequence_weight", "local_aux_weight")),
                 "baseline must not claim active geometry/fusion supervision")
    elif stage == "geometry":
        _require(loss["gate_weight"] == 0 and loss["residual_weight"] == 0, "geometry warmup disables gate/residual losses")
    optimizer, scheduler = cfg.get("optimizer"), cfg.get("scheduler")
    _require(isinstance(optimizer, dict) and optimizer.get("name") == "AdamW", "AdamW optimizer required")
    _require(optimizer.get("lr") == (1e-3 if stage == "baseline" else 1e-4), "unexpected phase learning rate")
    _require(optimizer.get("decoder_lr") == (1e-5 if stage == "fusion" else None), "unexpected decoder learning rate")
    decay = optimizer.get("weight_decay")
    _require(type(decay) in (int, float) and math.isfinite(decay) and decay >= 0, "invalid weight decay")
    _require(isinstance(scheduler, dict) and scheduler.get("name") == "StepLR", "explicit StepLR initial implementation required")
    _integer(scheduler.get("step_size"), "scheduler step_size", 1)
    _require(type(scheduler.get("gamma")) in (int, float) and 0 < scheduler["gamma"] <= 1, "invalid scheduler gamma")
    initialization = cfg.get("initialization")
    _require(isinstance(initialization, dict), "initialization provenance required")
    if stage == "baseline":
        _require(initialization == {"kind": "fresh"}, "formal baseline must start from fresh task weights")
    else:
        _require(initialization.get("kind") == "formal_parent" and initialization.get("parent_stage") == PARENT_STAGE[stage],
                 "formal stage requires the completed preceding formal stage")
        for key in ("parent_run_dir", "checkpoint_path"):
            _require(isinstance(initialization.get(key), str) and bool(initialization[key]), f"missing {key}")
        _sha(initialization.get("checkpoint_sha256"), "parent checkpoint SHA")
    model, mono = cfg.get("model"), cfg.get("mono")
    _require(isinstance(model, dict) and isinstance(mono, dict), "model and mono source configuration required")
    for key in ("vendor_root", "matcher_weights_path"):
        _require(isinstance(model.get(key), str) and Path(model[key]).is_absolute(), f"model.{key} must be absolute")
    for key in ("cache_dir", "fingerprint_path"):
        _require(isinstance(mono.get(key), str) and Path(mono[key]).is_absolute(), f"mono.{key} must be absolute")
    for key in ("matcher_weights_sha256", "vendor_source_sha256"):
        _sha(model.get(key), f"model.{key}")
    _sha(mono.get("fingerprint_sha256"), "mono.fingerprint_sha256")
    sources_verified = False
    if guard is not None:
        from .network import MOBILENET_V1_SHA256, VENDOR_SHA256
        from .monocular import _fingerprint
        _require(model["matcher_weights_sha256"] == MOBILENET_V1_SHA256, "matcher must use the fixed ImageNet weight, not task/reference weights")
        matcher = _inside_file(guard, model["matcher_weights_path"], "matcher")
        _require(sha256_file(matcher) == model["matcher_weights_sha256"], "matcher file SHA mismatch")
        vendor = guard.path(model["vendor_root"])
        actual = {name: sha256_file(_inside_file(guard, vendor / "models" / name, "vendor source")) for name in VENDOR_SHA256}
        _require(actual == VENDOR_SHA256 and canonical_hash(actual) == model["vendor_source_sha256"], "vendor source bytes/hash mismatch")
        fingerprint_path = _inside_file(guard, mono["fingerprint_path"], "monocular fingerprint")
        _require(sha256_file(fingerprint_path) == mono["fingerprint_sha256"], "monocular fingerprint file SHA mismatch")
        _fingerprint(_json(fingerprint_path), allow_cpu_checks=False)
        guard.path(mono["cache_dir"])
        sources_verified = True
    compatible = {key: value for key, value in cfg.items() if key not in {"execution", "run_id"}}
    return {"compatible": compatible, "config_sha256": canonical_hash(compatible),
            "execution": execution, "sources_verified": sources_verified}


def validate_manifest(manifest: Mapping, dataset_root: Path | None = None) -> dict:
    _require(isinstance(manifest, Mapping) and manifest.get("schema_version") == 1
             and manifest.get("dataset") == "TransCG", "expected full TransCG manifest")
    _require(not manifest.get("diagnostic_only", False) and manifest.get("mode", "formal") == "formal", "diagnostic manifest forbidden")
    _require(manifest.get("full_training_permitted", True) is not False, "manifest explicitly forbids full training")
    _require(manifest.get("seed") == 20261024 and manifest.get("split_scenes") == SPLITS, "manifest is not the frozen official 68/10/52 split")
    _require(manifest.get("original_split_scenes") == {"train": OFFICIAL_TRAIN_SCENES, "test": SPLITS["test"]}, "official split membership changed")
    _require(manifest.get("camera_depth_scales_m") == {"1": .001, "2": .00025}, "camera depth units changed")
    verification = manifest.get("verification", {})
    _require(verification.get("level") == "metadata_and_file_existence", "metadata-only manifest cannot launch formal training")
    _require(verification.get("metadata_files") == 131 and verification.get("metadata_snapshot_sha256") == METADATA_SNAPSHOT_SHA256,
             "manifest metadata identity is not the verified full dataset")
    receipts = verification.get("metadata_sha256")
    expected_names = {"metadata.json"} | {f"scene{i}/metadata.json" for i in range(1, 131)}
    _require(isinstance(receipts, dict) and set(receipts) == expected_names, "metadata receipt coverage incomplete")
    for digest in receipts.values():
        _sha(digest, "metadata SHA")
    _require(canonical_hash(receipts) == METADATA_SNAPSHOT_SHA256, "metadata receipt aggregate differs")
    if dataset_root is not None:
        _require(Path(manifest.get("dataset_root", "")).resolve(strict=True) == dataset_root.resolve(strict=True), "manifest dataset root mismatch")
    rows = manifest.get("samples")
    _require(isinstance(rows, list) and len(rows) == 57715, "formal manifest must contain 57,715 reference samples")
    ids, required = {}, set()
    totals = {split: [0, 0, 0, 0] for split in SPLITS}
    for row in rows:
        _require(isinstance(row, dict), "invalid sample row")
        split, scene, view, camera = (row.get(key) for key in ("split", "scene_id", "view_id", "reference_camera_id"))
        _require(split in SPLITS and type(scene) is int and scene in SPLITS[split]
                 and type(view) is int and 0 <= view < 240 and type(camera) is int and camera in (1, 2), "invalid sample membership")
        _require(row.get("mode") in {"paired", "single_view_fallback"} and row.get("scene_type") in {"isolated", "cluttered"}, "invalid sample condition")
        expected = _sample(split, scene, view, camera, row["mode"] == "paired", row["scene_type"])
        _require(row == expected, "sample paths, units, GT separation or calibration direction differ from fixed schema")
        _require(row["sample_id"] not in ids, "duplicate reference sample")
        ids[row["sample_id"]] = row
        count = totals[split]
        count[0] += 1
        count[1] += row["mode"] == "paired"
        count[1 + camera] += 1
        required.update(_file_paths(row))
    for row in rows:
        partner = ids.get(row["sample_id"][:-1] + str(3 - row["reference_camera_id"]))
        _require((partner is not None) == (row["mode"] == "paired"), "pair/single-view membership disagrees")
    for split, expected in COUNTS.items():
        _require(tuple(totals[split]) == expected, f"wrong full {split} reference/pair/camera counts")
        summary = manifest.get("summary", {}).get(split, {})
        _require(summary.get("samples") == expected[0] and summary.get("paired_reference_samples") == expected[1]
                 and summary.get("scene_count") == len(SPLITS[split]) and summary.get("single_view_fallback_samples") == expected[0] - expected[1],
                 f"{split} summary disagrees with actual rows")
    _require(verification.get("required_files_checked") == len(required), "full required-file coverage was not checked")
    return {split: {"samples": value[0], "paired_reference_samples": value[1], "scene_count": len(SPLITS[split])}
            for split, value in totals.items()}


def _validate_upload(guard: ResourceGuard, receipt_path: Path, dataset_root: Path) -> dict:
    receipt = _json(receipt_path)
    expected = {"schema_version": 1, "kind": "transcg_rsync_receipt", "source_server": "cvpr-server",
                "target_server": "cvpr-autodl", "source_root": TRANSFER_SOURCE,
                "target_host": TRANSFER_HOST, "target_port": TRANSFER_PORT}
    _require(all(receipt.get(key) == value for key, value in expected.items()), "upload receipt endpoint/source identity differs")
    _require(guard.path(receipt.get("dataset_root", "")) == dataset_root, "upload receipt binds another dataset root")
    _require(guard.path(receipt.get("target_datasets_root", "")) / "transcg" == dataset_root, "restricted rsync destination mapping differs")
    files = receipt.get("files", {})
    _require(set(files) == {"status", "checksum_diff", "checksum_stderr"}, "upload receipt requires original status/diff/stderr")
    paths = {}
    for name, info in files.items():
        _require(isinstance(info, dict), "malformed upload file receipt")
        path = _inside_file(guard, info.get("path"), name)
        _require(sha256_file(path) == _sha(info.get("sha256"), name), f"upload {name} bytes changed")
        paths[name] = path
    for name in ("checksum_diff", "checksum_stderr"):
        _require(paths[name].stat().st_size == 0, f"upload {name} is not empty")
    status = _json(paths["status"])
    _require(status.get("complete") is True and status.get("state") == "complete" and status.get("stage") == "verified"
             and "error" not in status, "uploader has no successful checksum receipt")
    _require(status.get("source") == TRANSFER_SOURCE and status.get("target") == f"root@{TRANSFER_HOST}:transcg/"
             and status.get("port") == TRANSFER_PORT, "uploader verified another source/target")
    _require(status.get("excludes") == ["/transcg-data-*.zip", "/transcg-info.zip"], "upload exclusion scope changed")
    _require(type(status.get("attempt")) is int and 1 <= status["attempt"] <= 3, "invalid uploader attempt")
    _integer(status.get("pid"), "uploader PID", 1)
    times = [status.get(name) for name in ("started_at", "finished_at", "updated_at")]
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in times)
             and 0 < times[0] <= times[1] <= times[2], "missing successful uploader completion timestamps")
    return receipt


def validate_gate_bundle(guard: ResourceGuard, manifest_path: Path | str,
                         upload_receipt_path: Path | str, calibration_path: Path | str) -> dict:
    manifest_file = _inside_file(guard, manifest_path, "manifest")
    upload_file = _inside_file(guard, upload_receipt_path, "upload receipt")
    calibration_file = _inside_file(guard, calibration_path, "calibration report")
    manifest = _json(manifest_file)
    dataset_root = guard.path("datasets/transcg")
    _require(dataset_root.is_dir(), "formal TransCG target is missing")
    validate_manifest(manifest, dataset_root)
    upload = _validate_upload(guard, upload_file, dataset_root)
    # Re-read just 131 metadata files, not hundreds of thousands of image files.
    # The separate completed rsync receipt supplies full content verification.
    regenerated = build_manifest(dataset_root, seed=20261024, verify_files=False)
    for key in ("samples", "summary", "split_scenes", "original_split_scenes"):
        _require(manifest[key] == regenerated[key], f"manifest no longer matches dataset metadata: {key}")
    _require(manifest["verification"]["metadata_sha256"] == regenerated["verification"]["metadata_sha256"], "dataset metadata bytes changed")
    manifest_sha = sha256_file(manifest_file)
    report = _json(calibration_file)
    fixed = {"schema_version": 1, "kind": "transcg_calibration_diagnostic", "protocol_version": "transcg_fixed_trainval_v1",
             "stage": "complete", "passed": True, "diagnostic_only": False, "manifest_sha256": manifest_sha,
             "test_samples_used": 0, "fitting_used": False}
    _require(all(report.get(key) == value for key, value in fixed.items()), "calibration gate is incomplete, diagnostic, failed or bound to another manifest")
    _require(Path(report.get("dataset_root", "")).resolve(strict=True) == dataset_root, "calibration dataset root differs")
    from .calibration import default_policy, evaluate_gate, select_pairs, validate_manifest as validate_calibration_manifest
    policy = default_policy()
    _require(report.get("policy") == policy and report.get("policy_sha256") == canonical_hash(policy), "calibration policy/thresholds changed")
    expected_sources = {name: sha256_file(_inside_file(guard, dataset_root / name, "calibration source")) for name in CALIBRATION_FILES}
    _require(report.get("calibration_sha256") == expected_sources, "camera calibration bytes changed")
    _require(report.get("root_metadata_sha256") == sha256_file(dataset_root / "metadata.json"), "calibration root metadata changed")
    selection = report.get("selection", {})
    _require(selection.get("method") == "per_scene_even_index_endpoints" and selection.get("views_per_scene") == 3
             and selection.get("seed") == 20261024
             and selection.get("expected_scene_ids") == {"train": SPLITS["train"], "val": SPLITS["val"]}, "calibration sample-selection protocol changed")
    calibrated_manifest = validate_calibration_manifest(manifest, _json(dataset_root / "metadata.json"), diagnostic_only=False)
    expected_selection, expected_pairs = select_pairs(calibrated_manifest, views_per_scene=3)
    _require(selection == expected_selection, "calibration views were not selected by the fixed manifest rule")
    samples = report.get("samples")
    _require(isinstance(samples, list) and len(samples) == 234, "calibration must retain all 234 fixed train/val pairs")
    for sample, selected_pair in zip(samples, expected_pairs):
        _require(all(sample.get(key) == value for key, value in selected_pair.items()), "calibration sample order/identity differs from fixed selection")
    rows = {row["sample_id"]: row for row in manifest["samples"]}
    seen = set()
    for sample in samples:
        refs = sample.get("reference_sample_ids", [])
        _require(isinstance(refs, list) and len(refs) == 2 and len(set(refs)) == 2 and all(value in rows for value in refs), "calibration references unknown samples")
        selected = [rows[value] for value in refs]
        scene, view, split = (sample.get(key) for key in ("scene_id", "view_id", "split"))
        _require(split in {"train", "val"} and all(row["mode"] == "paired" and row["scene_id"] == scene
                 and row["view_id"] == view and row["split"] == split for row in selected), "calibration pair membership/test leakage")
        _require({row["reference_camera_id"] for row in selected} == {1, 2} and (scene, view) not in seen, "duplicate calibration pair")
        seen.add((scene, view))
    summary = evaluate_gate(samples, selection, diagnostic_only=False)
    _require(summary.get("passed") is True, "calibration evidence does not pass independently recomputed gate")
    fingerprint = canonical_hash({"manifest_sha256": manifest_sha, "upload_receipt_sha256": sha256_file(upload_file),
                                  "calibration_report_sha256": sha256_file(calibration_file)})
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "dataset_root": dataset_root,
            "gate_fingerprint": fingerprint, "upload": upload, "calibration": report}


def validate_parent(guard: ResourceGuard, parent_info: Mapping, requested_stage: str,
                    manifest_sha: str, *, _seen: frozenset[str] = frozenset()) -> dict:
    _sha(manifest_sha, "manifest SHA")
    _require(requested_stage in BUDGETS and isinstance(parent_info, Mapping), "invalid requested stage/initialization")
    if requested_stage == "baseline":
        _require(dict(parent_info) == {"kind": "fresh"}, "baseline cannot initialize from a reference/diagnostic checkpoint")
        return {"checkpoint_path": None, "parent_config": None, "parent_registry": None}
    expected_stage = PARENT_STAGE[requested_stage]
    _require(parent_info.get("kind") == "formal_parent" and parent_info.get("parent_stage") == expected_stage,
             "wrong parent stage or nonformal initialization")
    run_dir = guard.path(parent_info.get("parent_run_dir", ""))
    _require(run_dir.parent == guard.path("experiment/remake_igev") and str(run_dir) not in _seen, "parent run path/cycle invalid")
    config = _json(_inside_file(guard, run_dir / "config.json", "parent config"))
    checked = validate_training_config(config, guard=guard)
    _require(config["stage"] == expected_stage and config["run_id"] == run_dir.name
             and config["manifest_sha256"] == manifest_sha, "parent formal phase/run/manifest identity differs")
    registry = _json(_inside_file(guard, run_dir / "registry.json", "parent registry"))
    _require(registry.get("format_version") == 1 and registry.get("run_id") == run_dir.name
             and registry.get("manifest_sha256") == manifest_sha and registry.get("config_sha256") == checked["config_sha256"],
             "parent registry/config hash identity differs")
    attempts = registry.get("attempts", [])
    _require(isinstance(attempts, list) and bool(attempts) and attempts[-1].get("status") == "completed"
             and registry.get("current_attempt") == attempts[-1].get("attempt_id"), "parent formal run is not complete")
    _require(all(isinstance(attempt, dict) and isinstance(attempt.get("attempt_id"), str)
                 and re.fullmatch(r"attempt-[0-9]{8}T[0-9]{6}-[a-f0-9]{8}", attempt["attempt_id"])
                 for attempt in attempts), "malformed parent attempt identity")
    _require(len({attempt["attempt_id"] for attempt in attempts}) == len(attempts), "duplicate parent attempt identity")
    progress = attempts[-1].get("progress", {})
    _require(type(progress.get("epoch")) is int and progress["epoch"] >= config["epochs"]
             and type(progress.get("validation_count")) is int and progress["validation_count"] >= 1,
             "parent completion lacks full phase budget/validation evidence")
    checkpoint = _inside_file(guard, parent_info.get("checkpoint_path"), "parent checkpoint")
    digest = _sha(parent_info.get("checkpoint_sha256"), "parent checkpoint SHA")
    _require(sha256_file(checkpoint) == digest, "parent checkpoint bytes differ")
    receipts = [registry.get(name) for name in ("best_checkpoint", "latest_checkpoint")]
    receipts += [attempt.get(name) for attempt in attempts for name in ("best_checkpoint", "latest_checkpoint")]
    matching = [receipt for receipt in receipts if isinstance(receipt, dict) and receipt.get("sha256") == digest
                and guard.path(receipt.get("path", "")) == checkpoint]
    _require(bool(matching), "checkpoint is not authenticated by parent registry")
    ids = {attempt.get("attempt_id") for attempt in attempts}
    _require(any(receipt.get("attempt_id") in ids and checkpoint.is_relative_to(run_dir / receipt["attempt_id"])
                 for receipt in matching), "checkpoint does not belong to a registered parent attempt")
    if expected_stage != "baseline":
        validate_parent(guard, config["initialization"], expected_stage, manifest_sha, _seen=_seen | {str(run_dir)})
    return {"checkpoint_path": checkpoint, "parent_config": config, "parent_registry": registry}
