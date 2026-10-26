"""Frozen, CUDA-only formal TransCG predictions with immutable resume artifacts.

The formal CLI currently accepts completed fusion runs only. This deliberately
holds test access until the B/G/F chain and validation selection are complete;
formal baseline/geometry comparison entry points are not implemented here.
The explicit diagnostic loop accepts train/val CPU fixtures, but never creates
a formal completion receipt. Only the declared privileged GT-mask mechanism
protocol is supported. GT depth stays in dataset targets, never model inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
from typing import Mapping

import numpy as np
import torch

from .dataset import TransCGPairsDataset, _safe_file
from .evaluation_artifacts import ArtifactError, PredictionStore
from .monocular import (CachedRelativeDepthProvider, FrozenDepthAnythingV2Small, RelativeDepthCache,
                        _fingerprint, grid_cache_fingerprint, read_original_rgb)
from .network import ReMakeGeometryNetwork
from .rescore import verify_formal_prediction_source
from .runtime import (ResourceGuard, RuntimeSafetyError, _atomic_write, _class_name, _progress_check,
                      assert_finite, canonical_hash, require_cuda_device, sha256_file)
from .train import _deployment
from .training_protocol import validate_manifest, validate_training_config


INPUT_KEYS = frozenset(("rgb_ref", "rgb_src", "raw_depth", "raw_valid", "relative_depth", "mask",
                        "K_ref", "K_src", "T_src_ref", "has_source", "source_valid"))


def _require(condition, message):
    if not condition:
        raise ArtifactError(message)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def validate_prediction_prior_fingerprint(value: Mapping, *, predictor_fingerprint: Mapping | None = None,
                                          diagnostic_only: bool = False) -> dict:
    """Keep formal test storage identical to the frozen 480x640 training grid.

    CPU fingerprints are accepted only by explicit diagnostic callers. Adding
    storage metadata to a native predictor is an identity comparison, never a
    change to its native-resolution inference or a device-mode conversion.
    """
    fingerprint = _fingerprint(value, allow_cpu_checks=diagnostic_only)
    _require(fingerprint.get("cache_storage", {}).get("output_hw") == [480, 640],
             "Prediction prior must use the frozen 480x640 training-grid cache")
    if predictor_fingerprint is not None:
        expected = grid_cache_fingerprint(predictor_fingerprint, output_hw=(480, 640))
        _require(expected == fingerprint, "Test DA producer differs from the training fingerprint")
    return fingerprint


def select_rows(manifest: Mapping, *, split: str = "test", diagnostic_only: bool = False) -> list[dict]:
    """Select metadata only, preserving the fixed manifest order and every fallback."""
    _require(isinstance(manifest, Mapping) and manifest.get("schema_version") == 1
             and manifest.get("dataset") == "TransCG", "Expected TransCG manifest")
    if diagnostic_only:
        _require(split in ("train", "val"), "Diagnostic predictions cannot access the test split")
    else:
        _require(split == "test", "Formal prediction split must be test")
        validate_manifest(manifest)
    all_rows = manifest.get("samples")
    _require(isinstance(all_rows, list) and all(isinstance(row, dict) for row in all_rows), "Malformed manifest rows")
    rows = [dict(row) for row in all_rows if row.get("split") == split]
    ids = [row.get("sample_id") for row in rows]
    _require(bool(rows) and all(isinstance(value, str) and value for value in ids)
             and len(ids) == len(set(ids)), "Missing or duplicate expected reference IDs")
    _require(all(row.get("mode") in ("paired", "single_view_fallback") for row in rows), "Invalid reference mode")
    if not diagnostic_only:
        _require(len(rows) == 23524 and sum(row["mode"] == "paired" for row in rows) == 23006
                 and sum(row["mode"] == "single_view_fallback" for row in rows) == 518,
                 "Formal test requires all 23,524 references, including 518 single-view fallbacks")
    return rows


def reference_provenance(guard: ResourceGuard, dataset_root, row, *, mask_protocol="gt",
                         relative_depth_provider=None, diagnostic_only=False) -> dict:
    """Hash original forward/scoring sources separately; never produce a GT input.

    A production cache is verified against the encoded RGB and its immutable
    predictor fingerprint. Diagnostic fixtures may use a plain RGB provider;
    they are labelled with mono_cache=None and cannot enter the formal loop.
    """
    _require(mask_protocol == "gt", "Automatic/predicted-mask provenance is not implemented")
    root = guard.path(dataset_root)
    _require(row.get("split") in (("train", "val") if diagnostic_only else ("test",)), "Wrong prediction data split")
    observation, supervision = row["inputs"], row["supervision"]
    forward_paths = {observation["rgb"], observation["raw_depth"], observation["intrinsics"], supervision["transparent_mask"]}
    source = observation.get("source")
    _require((source is not None) == (row["mode"] == "paired"), "Paired source declaration differs")
    if source is not None:
        forward_paths.update((source["rgb"], source["intrinsics"], source["reference_to_source_transform"]["path"]))
    _require(supervision["depth_gt"] not in forward_paths, "GT depth cannot be a forward source")
    forward = {relative: sha256_file(guard.path(_safe_file(root, relative, "forward input")))
               for relative in sorted(forward_paths)}
    scoring = {relative: sha256_file(guard.path(_safe_file(root, relative, "scoring provenance")))
               for relative in (observation["raw_depth"], supervision["depth_gt"], supervision["transparent_mask"])}
    for relative in (observation["raw_depth"], supervision["transparent_mask"]):
        _require(forward[relative] == scoring[relative], "Input changed while collecting scoring provenance")
    mono = None
    if isinstance(relative_depth_provider, CachedRelativeDepthProvider):
        cache = relative_depth_provider.cache
        _fingerprint(cache.fingerprint, allow_cpu_checks=diagnostic_only)
        if not diagnostic_only:
            validate_prediction_prior_fingerprint(cache.fingerprint)
        _require(cache.guard.root == guard.root, "Monocular cache belongs to another resource root")
        image = guard.path(_safe_file(root, observation["rgb"], "monocular RGB"))
        _require(relative_depth_provider.image_paths.get(row["sample_id"]) == str(image), "Monocular provider indexes another RGB")
        rgb, image_sha = read_original_rgb(image)
        _require(image_sha == forward[observation["rgb"]], "RGB changed while checking monocular cache")
        cached = cache.read(rgb, image_sha)
        mono = {"key": cached.key, "path": str(guard.path(cached.path)), "sha256": sha256_file(cached.path),
                "fingerprint_sha256": cache.fingerprint_sha256}
    else:
        _require(diagnostic_only and callable(relative_depth_provider), "Formal predictions require a frozen production CUDA prior cache")
    return {"sample_id": row["sample_id"], "split": row["split"], "scene_id": row["scene_id"],
            "view_id": row["view_id"], "reference_camera_id": row["reference_camera_id"], "mode": row["mode"],
            "mask_protocol": mask_protocol, "privileged_mask_input": True,
            "manifest_row_sha256": canonical_hash(row), "forward_input_sha256": forward,
            "scoring_input_sha256": scoring, "mono_cache": mono}


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Tensor bytes plus names/shapes/dtypes; independent of pickle serialization."""
    _require(isinstance(state, Mapping) and bool(state), "Expected a nonempty model tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        _require(isinstance(name, str) and isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
                 "Model state must contain named dense tensors")
        value = tensor.detach().cpu().contiguous()
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.nbytes.to_bytes(8, "little"))
        digest.update(memoryview(raw))
    return digest.hexdigest()


def load_selected_model_checkpoint(model, guard: ResourceGuard, checkpoint_path, *, expected_sha256: str,
                                   run_id: str, attempt_id: str, manifest_sha256: str,
                                   config_sha256: str, global_step: int) -> dict:
    """Load a trusted project best checkpoint after the formal-source gate.

    Callers must authenticate the completed run/registry before calling this
    helper. The same open file is hashed and deserialized, so a renamed path
    cannot swap bytes between those operations. No optimizer/RNG is restored.
    """
    path = guard.path(checkpoint_path)
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        _require(digest.hexdigest() == expected_sha256, "Selected checkpoint bytes differ from authenticated registry")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    expected = {"format_version": 1, "run_id": run_id, "attempt_id": attempt_id,
                "manifest_sha256": manifest_sha256, "config_sha256": config_sha256}
    _require(isinstance(payload, dict) and all(payload.get(key) == value for key, value in expected.items()),
             "Selected checkpoint run/attempt/manifest/config identity differs")
    _require(isinstance(payload.get("classes"), dict) and payload["classes"].get("model") == _class_name(model),
             "Selected checkpoint model class differs")
    assert_finite(payload, "selected prediction checkpoint")
    _progress_check(payload["progress"], payload["data_state"])
    progress = payload["progress"]
    best = progress.get("best_metrics", {})
    metric = best.get("transparent_mae_mm") if isinstance(best, dict) else None
    _require(type(global_step) is int and global_step >= 0 and progress["global_step"] == global_step
             and progress.get("phase") == "ready" and isinstance(best, dict)
             and type(best.get("epoch")) is int and best["epoch"] >= 0
             and best["epoch"] + 1 == progress["epoch"] and best.get("global_step") == global_step
             and type(metric) in (int, float) and math.isfinite(metric) and metric >= 0,
             "Selected checkpoint does not represent its recorded validation best")
    state = payload.get("states", {}).get("model")
    current = model.state_dict()
    _require(isinstance(state, Mapping) and current.keys() == state.keys()
             and all(isinstance(state[name], torch.Tensor) and state[name].shape == current[name].shape
                     and state[name].dtype == current[name].dtype for name in current),
             "Selected model tensor keys/shapes/dtypes differ")
    state_sha = model_state_sha256(state)
    model.load_state_dict(state, strict=True)
    return {"model_state_sha256": state_sha, "progress": progress, "attempt_id": attempt_id}


def predict_references(model, dataset, store: PredictionStore, *, device, iterations=8,
                       diagnostic_only=False, allow_cpu_checks=False, on_progress=None, should_stop=None) -> dict:
    """Microbatch-one inference; caller holds store.writer_lock().

    on_progress receives a fresh progress dict after each verified reference.
    should_stop is a zero-argument callback checked at reference boundaries.
    CPU requires both diagnostic_only and allow_cpu_checks; diagnostic data and
    artifacts must be train/val. This loop never writes a completion receipt.
    """
    _require(type(iterations) is int and iterations > 0, "iterations must be positive")
    requested = torch.device(device)
    if requested.type == "cpu":
        _require(diagnostic_only and allow_cpu_checks, "CPU prediction is allowed only for explicit diagnostic checks")
    else:
        _require(requested.type == "cuda", "Only CUDA or explicit diagnostic CPU is supported")
        requested = require_cuda_device(requested)
        if requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
    expected_mode = "diagnostic" if diagnostic_only else "formal"
    protocol = store.protocol
    _require(protocol.get("kind") == "remake_igev_predictions" and protocol.get("mode") == expected_mode
             and protocol.get("output_hw") == [480, 640], "Prediction store mode/grid differs")
    _require(dataset.mask_protocol == "gt" and protocol.get("mask_protocol") == "gt", "Predicted-mask inference is not implemented")
    _require(dataset.split in (("train", "val") if diagnostic_only else ("test",))
             and protocol.get("split") == dataset.split, "Prediction split differs or diagnostics access test")
    if not diagnostic_only:
        _require(isinstance(dataset, TransCGPairsDataset), "Formal prediction requires the canonical dataset adapter")
        _require(protocol.get("stage") == "fusion" and getattr(model, "stage", None) == "fusion",
                 "Formal test entry currently requires the completed fusion stage")
    rows = dataset.rows
    ids = [row["sample_id"] for row in rows]
    _require(bool(ids) and len(ids) == len(set(ids)) and protocol.get("expected_sample_ids") == ids,
             "Prediction coverage/order differs from dataset references")
    if not diagnostic_only:
        _require(len(rows) == 23524 and sum(row["mode"] == "paired" for row in rows) == 23006
                 and sum(row["mode"] == "single_view_fallback" for row in rows) == 518,
                 "Formal test reference coverage differs")
        _require(protocol.get("iterations") == iterations, "Formal iteration budget differs")
    root = store.guard.path(dataset.root)
    _require(protocol.get("dataset_root") == str(root), "Prediction dataset root differs")
    _require(all(tensor.device == requested for tensor in list(model.parameters()) + list(model.buffers())),
             "Model is not entirely on the explicitly requested device")
    model.requires_grad_(False).eval()
    progress = {"mode": expected_mode, "complete": False, "expected": len(ids), "completed": 0,
                "written": 0, "reused": 0, "next_sample_index": 0}
    with torch.inference_mode(), torch.autocast(device_type=requested.type, enabled=False):
        for index, row in enumerate(rows):
            if should_stop is not None and should_stop():
                return dict(progress)
            metadata = reference_provenance(store.guard, root, row, mask_protocol=dataset.mask_protocol,
                                            relative_depth_provider=dataset.relative_depth_provider,
                                            diagnostic_only=diagnostic_only)
            if store.contains(row["sample_id"]):
                saved = store.read(row["sample_id"])
                _require(saved["record"]["metadata"] == metadata, "Existing prediction input/scoring provenance changed; use a new evaluation")
                progress["reused"] += 1
            else:
                sample = dataset[index]
                sample_metadata = sample.get("metadata", {})
                for key in ("sample_id", "split", "scene_id", "view_id", "reference_camera_id", "mode", "mask_protocol"):
                    _require(sample_metadata.get(key) == metadata[key], f"Dataset reference index differs: {key}")
                inputs = sample.get("inputs")
                _require(isinstance(inputs, Mapping) and INPUT_KEYS <= set(inputs)
                         and set(inputs) <= INPUT_KEYS | {"reference_valid"}, "Unexpected model input keys; GT/targets are forbidden")
                _require(all(isinstance(value, torch.Tensor) for value in inputs.values()), "Dataset model inputs must be tensors")
                batched = {key: value.unsqueeze(0).to(requested) for key, value in inputs.items()}
                output = model(batched, iterations=iterations)
                prediction = output.depth
                _require(isinstance(prediction, torch.Tensor) and prediction.shape == (1, 1, 480, 640)
                         and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all())
                         and bool((prediction >= 0).all()), "Model output must be finite nonnegative FP32 optical-z depth on the evaluation grid")
                baseline = output.baseline_depth
                _require(isinstance(baseline, torch.Tensor) and baseline.shape == prediction.shape
                         and baseline.dtype == torch.float32 and baseline.device == prediction.device
                         and bool(torch.isfinite(baseline).all()) and bool((baseline >= 0).all()),
                         "Baseline output must be finite nonnegative FP32 depth with matching shape/device")
                geometry = output.geometry
                _require(diagnostic_only or geometry is not None, "Formal fusion output must declare geometry support")
                support = None
                if geometry is not None:
                    valid = geometry.valid
                    _require(isinstance(valid, torch.Tensor) and valid.dtype == torch.bool
                             and valid.shape == prediction.shape and valid.device == prediction.device,
                             "Geometry support must be bool B1HW output validity on the same device")
                    _require(torch.equal(prediction[~valid], baseline[~valid]), "Unsupported geometry must preserve the exact baseline prediction")
                    if row["mode"] == "single_view_fallback":
                        _require(not bool(valid.any()), "Single-view fallback unexpectedly claims geometric support")
                    support = valid[0, 0].cpu().numpy().copy()
                if geometry is None or row["mode"] == "single_view_fallback":
                    _require(torch.equal(prediction, baseline), "Baseline/single-view output must be the exact baseline prediction")
                after = reference_provenance(store.guard, root, row, mask_protocol=dataset.mask_protocol,
                                             relative_depth_provider=dataset.relative_depth_provider,
                                             diagnostic_only=diagnostic_only)
                _require(after == metadata, "Reference sources changed during inference; prediction not published")
                store.write(row["sample_id"], prediction[0, 0].float().cpu().numpy().copy(),
                            geometry_support=support, metadata=metadata)
                progress["written"] += 1
            progress.update(completed=index + 1, next_sample_index=index + 1)
            if on_progress is not None:
                on_progress(dict(progress))
    store.assert_complete(ids)
    progress["complete"] = True
    return dict(progress)


@contextmanager
def _completed_source_lock(guard, run_dir):
    _require(run_dir.parent == guard.path("experiment/remake_igev"), "Source run must belong to this task registry")
    lock_path = guard.path(run_dir / "run.lock")
    _require(lock_path.is_file(), "Formal source run lock is missing")
    with lock_path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError("Source training run is still owned by an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(*, resource_root, project_root, run_dir, prediction_dir, mask_protocol="gt") -> dict:
    """Authenticate/freeze a complete B/G/F source, then cache and predict test."""
    _require(mask_protocol == "gt", "Only the explicitly privileged GT-mask mechanism protocol is implemented")
    guard = ResourceGuard(resource_root, minimum_free_bytes=80 * 2**30)
    project, source_run = guard.path(project_root), guard.path(run_dir)
    with _completed_source_lock(guard, source_run):
        config_path = guard.path(source_run / "config.json")
        registry_path = guard.path(source_run / "registry.json")
        config, registry = _json(config_path), _json(registry_path)
        _require(config.get("stage") == "fusion", "Formal test currently requires completed fusion; baseline/geometry comparison entry points are pending")
        checked = validate_training_config(config, guard=guard)
        manifest_path = guard.path(config["manifest_path"])
        best = registry.get("best_checkpoint", {})
        _require(isinstance(best, dict), "Formal source has no selected best checkpoint")
        protocol = {"schema_version": 1, "kind": "remake_igev_predictions", "mode": "formal", "split": "test",
                    "output_hw": [480, 640], "mask_protocol": mask_protocol, "privileged_mask_input": True,
                    "run_dir": str(source_run), "run_config_sha256": checked["config_sha256"],
                    "manifest_sha256": sha256_file(manifest_path), "stage": config["stage"],
                    "checkpoint_path": str(guard.path(best.get("path", ""))), "checkpoint_sha256": best.get("sha256"),
                    "iterations": config["iterations"], "precision": "fp32", "microbatch": 1,
                    "GT_depth_usage": "scoring_provenance_only_never_forward_input", "crop": None}
        # This validates completion, val-selected best, formal parent chain,
        # complete upload/manifest and camera gate before any test image/cache.
        verified = verify_formal_prediction_source(guard, protocol, manifest_path)
        _require(verified == config, "Source configuration changed during verification")
        revision = _deployment(guard, project)
        deployment_path = guard.path(project / "deployment.json")
        deployment = _json(deployment_path)
        manifest = _json(manifest_path)
        rows = select_rows(manifest)
        root = guard.path(manifest["dataset_root"])
        fingerprint_path = guard.path(config["mono"]["fingerprint_path"])
        fingerprint = validate_prediction_prior_fingerprint(_json(fingerprint_path))
        _require(sha256_file(fingerprint_path) == config["mono"]["fingerprint_sha256"], "Production monocular fingerprint changed")
        best_attempts = [item for item in registry["attempts"] if item["attempt_id"] == best["attempt_id"]]
        _require(len(best_attempts) == 1 and isinstance(best_attempts[0].get("source_revision"), str), "Best checkpoint lacks training source identity")
        protocol.update(dataset_root=str(root), expected_sample_ids=[row["sample_id"] for row in rows],
                        source_revision=revision, source_files_sha256=deployment["source_files_sha256"],
                        deployment_sha256=sha256_file(deployment_path),
                        training_source_revision=best_attempts[0]["source_revision"],
                        run_config_file_sha256=sha256_file(config_path), registry_sha256=sha256_file(registry_path),
                        mono_fingerprint_file_sha256=config["mono"]["fingerprint_sha256"],
                        mono_cache_fingerprint_sha256=canonical_hash(fingerprint))
        device = require_cuda_device("cuda:0")
        model = ReMakeGeometryNetwork(vendor_root=guard.path(config["model"]["vendor_root"]),
                                      matcher_weights_path=guard.path(config["model"]["matcher_weights_path"]),
                                      stage=config["stage"], geometry_options=config["geometry"], freeze_bn_stats=True)
        loaded = load_selected_model_checkpoint(model, guard, protocol["checkpoint_path"],
                                               expected_sha256=protocol["checkpoint_sha256"], run_id=config["run_id"],
                                               attempt_id=best["attempt_id"], manifest_sha256=protocol["manifest_sha256"],
                                               config_sha256=checked["config_sha256"], global_step=best["global_step"])
        protocol["model_state_sha256"] = loaded["model_state_sha256"]
        store = PredictionStore(guard, prediction_dir, protocol=protocol, create=True)
        stopped = False
        def request_stop(signum, frame):
            nonlocal stopped
            stopped = True
        handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
        for number in handlers:
            signal.signal(number, request_stop)
        try:
            with store.writer_lock():
                cache = RelativeDepthCache(guard, config["mono"]["cache_dir"], fingerprint=fingerprint)
                # The fixed DA predictor is constructed only after the completed
                # source and artifact protocol are frozen. Cache misses are
                # populated here in the main process, never in dataset workers.
                predictor = FrozenDepthAnythingV2Small(
                    guard.path(project / "vendor/depth_anything_v2/src"),
                    guard.path(project / "weights/depth_anything/depth_anything_v2_vits.pth"),
                    device=device, input_size=fingerprint["input_size"])
                validate_prediction_prior_fingerprint(fingerprint, predictor_fingerprint=predictor.fingerprint)
                image_paths = {}
                for index, row in enumerate(rows):
                    if stopped:
                        return {"mode": "formal", "complete": False, "phase": "monocular_cache", "cache_references_checked": index}
                    image = guard.path(_safe_file(root, row["inputs"]["rgb"], "test RGB cache source"))
                    image_paths[row["sample_id"]] = image
                    result = cache.populate_file(image, predictor)
                    if not result.hit or (index + 1) % 100 == 0:
                        print(json.dumps({"event": "test_cache", "checked": index + 1, "expected": len(rows), "cache_hit": result.hit}), flush=True)
                del predictor
                model.to(device).requires_grad_(False).eval()
                dataset = TransCGPairsDataset(manifest, "test", mask_protocol="gt",
                                              relative_depth_provider=cache.provider(image_paths), output_size=(480, 640))
                def progress(value):
                    if value["completed"] % 100 == 0:
                        print(json.dumps({"event": "prediction_progress", **value}), flush=True)
                result = predict_references(model, dataset, store, device=device, iterations=config["iterations"],
                                            on_progress=progress, should_stop=lambda: stopped)
                if result["complete"]:
                    _require(sha256_file(config_path) == protocol["run_config_file_sha256"]
                             and sha256_file(registry_path) == protocol["registry_sha256"]
                             and sha256_file(manifest_path) == protocol["manifest_sha256"]
                             and sha256_file(deployment_path) == protocol["deployment_sha256"], "Frozen formal source changed during prediction")
                    _require(_deployment(guard, project) == revision, "Prediction source deployment changed")
                    _require(model_state_sha256(model.state_dict()) == protocol["model_state_sha256"], "Model state mutated during eval")
                    completion = {"schema_version": 1, "kind": "remake_igev_formal_prediction_completion",
                                  "mode": "formal", "device_type": "cuda", "complete": True,
                                  "protocol_sha256": store.protocol_sha256, "reference_samples": len(rows),
                                  "paired_reference_samples": 23006, "single_view_fallback_samples": 518,
                                  "checkpoint_sha256": protocol["checkpoint_sha256"],
                                  "model_state_sha256": protocol["model_state_sha256"], "metrics_computed": False}
                    receipt = guard.path(store.directory / "completion.json")
                    if receipt.exists():
                        _require(_json(receipt) == completion, "Existing prediction completion receipt differs")
                    else:
                        raw = json.dumps(completion, sort_keys=True, allow_nan=False).encode("utf-8")
                        _atomic_write(guard, receipt, lambda stream: stream.write(raw), replace=False)
                return result
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resource-root", "project-root", "run-dir", "prediction-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mask-protocol", choices=("gt", "predicted"), default="gt")
    args = parser.parse_args()
    result = run(resource_root=args.resource_root, project_root=args.project_root, run_dir=args.run_dir,
                 prediction_dir=args.prediction_dir, mask_protocol=args.mask_protocol)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
