"""Linux training ownership and recoverable checkpoints on an explicit data disk.

Checkpoint inputs must be trusted files produced by this project. Save only at
optimizer-step boundaries after zero_grad: pending accumulated gradients and
DataLoader worker/prefetch RNG are deliberately not serialized. Reproducible
loading requires deterministic per-sample augmentation and a resumable sampler.
This module never selects a CPU fallback for training.
"""
from __future__ import annotations

import datetime as dt
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
import torch


class RuntimeSafetyError(RuntimeError):
    """Unsafe storage, ownership, state, or resume request."""


class RunLockedError(RuntimeSafetyError):
    """Another process holds the run's kernel lock."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeSafetyError(f"{name} must be a lowercase SHA-256 digest")
    return value


class ResourceGuard:
    """Reject missing mounts, root-filesystem storage and escaping paths.

    The root must already exist. A captured st_dev is checked before every
    write so a missing/replaced mount cannot silently redirect to '/'. On bzl,
    the caller must additionally perform the project's required UUID preflight.
    """

    def __init__(self, resource_root: Path | str, minimum_free_bytes: int = 0):
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes must be a nonnegative integer")
        self.requested_root = Path(resource_root).absolute()
        self.root = self.requested_root.resolve(strict=True)
        self.device = self.root.stat().st_dev
        self.minimum_free_bytes = minimum_free_bytes
        self.validate()

    def validate(self) -> None:
        root = self.requested_root.resolve(strict=True)
        if root != self.root or not root.is_dir() or root.stat().st_dev != self.device:
            raise RuntimeSafetyError("resource root or mounted filesystem changed")
        if self.device == Path("/").stat().st_dev:
            raise RuntimeSafetyError("resource root is on the system filesystem")
        if os.statvfs(root).f_flag & os.ST_RDONLY or not os.access(root, os.W_OK):
            raise RuntimeSafetyError("resource root is read-only or not writable")
        if shutil.disk_usage(root).free < self.minimum_free_bytes:
            raise RuntimeSafetyError("resource root has insufficient free bytes")

    def path(self, value: Path | str) -> Path:
        self.validate()
        value = Path(value)
        target = (value if value.is_absolute() else self.root / value).resolve()
        if not target.is_relative_to(self.root):
            raise RuntimeSafetyError(f"path escapes resource root: {value}")
        ancestor = target
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.stat().st_dev != self.device:
            raise RuntimeSafetyError("output resolves to a different filesystem")
        return target

    def mkdir(self, value: Path | str) -> Path:
        path = self.path(value)
        path.mkdir(parents=True, exist_ok=True)
        return self.path(path)


def require_cuda_device(device: str | torch.device = "cuda") -> torch.device:
    result = torch.device(device)
    if result.type != "cuda":
        raise RuntimeSafetyError("training requires CUDA; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeSafetyError("CUDA unavailable; do not start or fall back to CPU")
    # Availability through NVML alone is insufficient; initialization may fail.
    torch.cuda.init()
    if result.index is not None and result.index >= torch.cuda.device_count():
        raise RuntimeSafetyError("requested CUDA device does not exist")
    return result


def process_identity() -> dict:
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return {
        "pid": pid, "start_ticks": stat.rsplit(")", 1)[1].split()[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
        "hostname": socket.gethostname(), "pgid": os.getpgrp(),
        "argv": list(sys.argv), "executable": sys.executable,
        "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").rstrip("\0").split("\0"),
    }


def process_matches(owner: dict) -> bool:
    try:
        if (owner["hostname"] != socket.gethostname()
                or owner["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()):
            return False
        pid = owner["pid"]
        if type(pid) is not int or pid <= 0:
            return False
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").rstrip("\0").split("\0")
        return (fields[0] != "Z" and fields[19] == owner["start_ticks"]
                and os.getpgid(pid) == owner["pgid"] and command == owner["cmdline"])
    except (KeyError, OSError, IndexError, TypeError):
        return False


def _atomic_write(guard: ResourceGuard, destination: Path, writer, *, replace: bool) -> None:
    destination = guard.path(destination)
    guard.mkdir(destination.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                     dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        guard.path(destination)
        if replace:
            os.replace(temporary, destination)
        else:
            # Immutable checkpoint: exclusive hard-link publication is atomic
            # and never overwrites a previous good checkpoint, even on failure.
            os.link(temporary, destination)
            temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(guard: ResourceGuard, destination: Path, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _atomic_write(guard, destination, lambda stream: stream.write(raw), replace=True)


def assert_finite(value: Any, name: str = "state") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise RuntimeSafetyError(f"nonfinite tensor in {name}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise RuntimeSafetyError(f"nonfinite array in {name}")
    elif isinstance(value, (float, complex, np.floating, np.complexfloating)):
        if not bool(np.isfinite(value)):
            raise RuntimeSafetyError(f"nonfinite value in {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_finite(item, f"{name}[{index}]")


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        if isinstance(value, OrderedDict):
            result = OrderedDict((key, _cpu_copy(item)) for key, item in value.items())
            if hasattr(value, "_metadata"):
                result._metadata = _cpu_copy(value._metadata)
            return result
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def capture_rng() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _class_name(component) -> str | None:
    return None if component is None else f"{type(component).__module__}.{type(component).__qualname__}"


def _progress_check(progress: dict, data_state: dict) -> None:
    for key in ("epoch", "global_step", "next_batch", "accumulation_position"):
        if type(progress.get(key)) is not int or progress[key] < 0:
            raise RuntimeSafetyError(f"invalid progress.{key}")
    if progress["accumulation_position"] != 0:
        raise RuntimeSafetyError("checkpoint only at completed optimizer-step boundaries")
    if not isinstance(progress.get("best_metrics"), dict):
        raise RuntimeSafetyError("progress.best_metrics must be a finite metric dictionary")
    for key in ("sampler_epoch", "sampler_seed", "next_sample_index", "dataset_size"):
        if type(data_state.get(key)) is not int or data_state[key] < 0:
            raise RuntimeSafetyError(f"invalid data_state.{key}")
    if (data_state["sampler_epoch"] != progress["epoch"]
            or data_state["next_sample_index"] > data_state["dataset_size"]):
        raise RuntimeSafetyError("sampler position and progress disagree")
    assert_finite(progress, "progress")
    assert_finite(data_state, "data_state")


def save_checkpoint(path: Path, *, guard: ResourceGuard, run_id: str, attempt_id: str,
                    manifest_sha256: str, config_sha256: str, model, optimizer,
                    scheduler, scaler, progress: dict, data_state: dict) -> Path:
    """Publish one immutable, complete optimizer-step checkpoint; never replace."""
    _sha(manifest_sha256, "manifest_sha256")
    _sha(config_sha256, "config_sha256")
    _progress_check(progress, data_state)
    if any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
           for parameter in model.parameters()):
        raise RuntimeSafetyError("zero_grad before saving: pending gradients are not checkpointed")
    components = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    payload = {
        "format_version": 1, "run_id": run_id, "attempt_id": attempt_id,
        "manifest_sha256": manifest_sha256, "config_sha256": config_sha256,
        "created_at": _now(), "classes": {key: _class_name(item) for key, item in components.items()},
        "states": {key: _cpu_copy(item.state_dict()) if item is not None else None
                   for key, item in components.items()},
        "rng": capture_rng(), "progress": _cpu_copy(progress), "data_state": _cpu_copy(data_state),
    }
    assert_finite(payload)
    destination = guard.path(path)
    _atomic_write(guard, destination, lambda stream: torch.save(payload, stream), replace=False)
    return destination


def load_checkpoint(path: Path, *, guard: ResourceGuard, expected_run_id: str,
                    manifest_sha256: str, config_sha256: str, model, optimizer,
                    scheduler, scaler, restore_random_state: bool = True) -> dict:
    """Validate trusted project checkpoint before loading; never use for uploads."""
    destination = guard.path(path)
    _sha(manifest_sha256, "manifest_sha256")
    _sha(config_sha256, "config_sha256")
    # Full RNG/NumPy state needs pickle; only locally generated trusted files.
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    expected = {"format_version": 1, "run_id": expected_run_id,
                "manifest_sha256": manifest_sha256, "config_sha256": config_sha256}
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise RuntimeSafetyError("checkpoint version, run, manifest or config mismatch")
    assert_finite(payload)
    _progress_check(payload["progress"], payload["data_state"])
    components = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    if payload["classes"] != {key: _class_name(item) for key, item in components.items()}:
        raise RuntimeSafetyError("checkpoint component types differ")
    states = payload["states"]
    current = model.state_dict()
    if (current.keys() != states["model"].keys()
            or any(current[k].shape != states["model"][k].shape
                   or current[k].dtype != states["model"][k].dtype for k in current)):
        raise RuntimeSafetyError("model state keys, shapes or dtypes differ")
    old_groups = states["optimizer"]["param_groups"]
    if (len(old_groups) != len(optimizer.param_groups)
            or any(len(old["params"]) != len(new["params"])
                   for old, new in zip(old_groups, optimizer.param_groups))):
        raise RuntimeSafetyError("optimizer parameter groups differ")
    for old, new in zip(old_groups, optimizer.param_groups):
        for key, parameter in zip(old["params"], new["params"]):
            for item in states["optimizer"]["state"].get(key, {}).values():
                if isinstance(item, torch.Tensor) and item.numel() != 1 and item.shape != parameter.shape:
                    raise RuntimeSafetyError("optimizer state shape differs from parameter")
    rng = payload["rng"]
    if restore_random_state:
        # Validate RNG using private generators; reject errors before mutation.
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"])
        if rng["cuda"]:
            require_cuda_device()
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise RuntimeSafetyError("CUDA RNG device count differs")
            for state in rng["cuda"]:
                if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
                    raise RuntimeSafetyError("malformed CUDA RNG state")
    for name, component in components.items():
        if component is not None:
            component.load_state_dict(states[name])
    if restore_random_state:
        restore_rng(rng)
    return {key: payload[key] for key in ("progress", "data_state", "run_id", "attempt_id", "created_at")}


class RunRegistry:
    """One trainer process per run. No killing, automatic retry, or CPU fallback.

    Hold this context in the actual trainer, not its SSH launcher. Normal context
    exit records 'stopped'; call finish('completed') only after actual completion.
    An interrupted attempt is preserved when a new process acquires a free lock.
    config_sha256 must bind the algorithm/protocol, including effective batch.
    Physical batch and gradient accumulation belong in execution_config and may
    change for OOM recovery only while preserving the compatible effective batch.
    This class never weakens hash matching or decides such compatibility itself.
    """

    def __init__(self, guard: ResourceGuard, run_id: str, *, manifest_sha256: str,
                 config_sha256: str, execution_config: dict | None = None,
                 source_revision: str | None = None):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
            raise ValueError("invalid run_id")
        self.guard, self.run_id = guard, run_id
        self.manifest_sha256 = _sha(manifest_sha256, "manifest_sha256")
        self.config_sha256 = _sha(config_sha256, "config_sha256")
        if execution_config is not None and not isinstance(execution_config, dict):
            raise TypeError("execution_config must be a JSON dictionary")
        if source_revision is not None and (not isinstance(source_revision, str) or not source_revision.strip()):
            raise ValueError("source_revision must be a nonempty revision/diff identifier")
        self.execution_config = json.loads(json.dumps(execution_config or {}, allow_nan=False))
        self.source_revision = source_revision
        self.run_dir = guard.path(Path("experiment/remake_igev") / run_id)
        self.registry_path = self.run_dir / "registry.json"
        self._lock = None
        self.owner = None
        self.attempt_id = None
        self.attempt_dir = None
        self.registry = None

    def __enter__(self):
        if self._lock is not None:
            raise RuntimeSafetyError("registry context is already active")
        self.guard.mkdir(self.run_dir)
        self._lock = self.guard.path(self.run_dir / "run.lock").open("a+b")
        try:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunLockedError(f"run is already locked: {self.run_id}") from exc
            self.owner = process_identity()
            registry_path = self.guard.path(self.registry_path)
            if registry_path.exists():
                self.registry = json.loads(registry_path.read_text(encoding="utf-8"))
                for key, expected in (("format_version", 1), ("run_id", self.run_id),
                                      ("manifest_sha256", self.manifest_sha256),
                                      ("config_sha256", self.config_sha256)):
                    if self.registry.get(key) != expected:
                        raise RuntimeSafetyError(f"run registry {key} mismatch; use a separate run")
                if self.registry["attempts"] and self.registry["attempts"][-1]["status"] == "completed":
                    raise RuntimeSafetyError("run already completed; use a new run_id for new work")
                for attempt in self.registry["attempts"]:
                    if attempt["status"] == "running":
                        if process_matches(attempt["owner"]):
                            raise RuntimeSafetyError("recorded trainer is still alive despite an available lock")
                        attempt.update(status="interrupted", ended_at=_now())
            else:
                self.registry = {"format_version": 1, "run_id": self.run_id,
                                 "manifest_sha256": self.manifest_sha256,
                                 "config_sha256": self.config_sha256,
                                 "attempts": [], "latest_checkpoint": None}
            self.attempt_id = "attempt-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            self.attempt_dir = self.guard.path(self.run_dir / self.attempt_id)
            self.attempt_dir.mkdir(exist_ok=False)
            self.registry["attempts"].append({"attempt_id": self.attempt_id, "owner": self.owner,
                                              "status": "running", "started_at": _now(),
                                              "execution_config": self.execution_config,
                                              "source_revision": self.source_revision,
                                              "event_count": 0})
            self.registry["current_attempt"] = self.attempt_id
            self._flush()
            return self
        except BaseException:
            self._release()
            raise

    def assert_owned(self) -> None:
        if self._lock is None or self._lock.closed or self.owner != process_identity():
            raise RuntimeSafetyError("run lock belongs to another process or has been released")

    def _flush(self) -> None:
        self.assert_owned()
        self.registry["updated_at"] = _now()
        _write_json(self.guard, self.registry_path, self.registry)

    def save_checkpoint(self, *, model, optimizer, scheduler, scaler,
                        progress: dict, data_state: dict) -> Path:
        self.assert_owned()
        if self.registry["attempts"][-1]["status"] != "running":
            raise RuntimeSafetyError("cannot checkpoint a finished attempt")
        path = self.attempt_dir / "checkpoints" / f"step-{progress['global_step']:09d}-{uuid.uuid4().hex[:8]}.pt"
        path = save_checkpoint(path, guard=self.guard, run_id=self.run_id, attempt_id=self.attempt_id,
                               manifest_sha256=self.manifest_sha256, config_sha256=self.config_sha256,
                               model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                               progress=progress, data_state=data_state)
        receipt = {"path": str(path.relative_to(self.guard.root)), "sha256": sha256_file(path),
                   "global_step": progress["global_step"], "attempt_id": self.attempt_id,
                   "bytes": path.stat().st_size}
        previous = self.registry["latest_checkpoint"]
        self.registry["previous_checkpoint"] = previous
        self.registry["latest_checkpoint"] = receipt
        attempt = self.registry["attempts"][-1]
        attempt["previous_checkpoint"] = previous
        attempt["latest_checkpoint"] = receipt
        attempt.setdefault("checkpoint_history", []).append(dict(receipt))
        self._flush()
        return path

    def publish_best_checkpoint(self) -> None:
        """Pin the already committed latest checkpoint before any retention pass."""
        self.latest_checkpoint()  # Authenticate bytes before publishing the pin.
        receipt = self.registry["latest_checkpoint"]
        if receipt is None or self.registry["attempts"][-1]["status"] != "running":
            raise RuntimeSafetyError("best checkpoint requires a live attempt and a saved checkpoint")
        self.registry["best_checkpoint"] = dict(receipt)
        self.registry["attempts"][-1]["best_checkpoint"] = dict(receipt)
        self._flush()

    def prune_checkpoints(self) -> dict:
        """Retain latest, previous, and every registry best/attempt recovery pin.

        Explicit opt-in for a running trainer, after save and best publication.
        Only checkpoint receipts created by this current attempt are eligible;
        never glob directories, touch earlier/failed attempts, or infer ownership
        from a filename. The deletion journal is durable before each named file
        can be unlinked. A failed save/registry publication forbids pruning.
        """
        self.assert_owned()
        attempt = self.registry["attempts"][-1]
        if attempt["status"] != "running":
            raise RuntimeSafetyError("checkpoint retention never cleans a finished/failed attempt")
        persisted = json.loads(self.guard.path(self.registry_path).read_text(encoding="utf-8"))
        if persisted != self.registry:
            raise RuntimeSafetyError("checkpoint registry has unpublished changes; refuse retention")
        checkpoint_directory = self.attempt_dir / "checkpoints"
        directory = self.guard.path(checkpoint_directory)
        if (self.attempt_dir.is_symlink() or checkpoint_directory.is_symlink()
                or directory.parent != self.attempt_dir):
            raise RuntimeSafetyError("checkpoint directory was replaced or redirected; refuse retention")
        pins = set()
        for owner in [self.registry, *self.registry["attempts"]]:
            for name in ("latest_checkpoint", "previous_checkpoint", "best_checkpoint"):
                receipt = owner.get(name)
                if receipt is not None:
                    pins.add(str(self.guard.path(receipt["path"])))
        # A usable previous recovery point is a requirement, not just a filename.
        for name in ("latest_checkpoint", "previous_checkpoint", "best_checkpoint"):
            receipt = self.registry.get(name)
            if receipt is not None:
                target = self.guard.path(receipt["path"])
                if not target.is_file() or sha256_file(target) != receipt["sha256"]:
                    raise RuntimeSafetyError(f"protected {name} is missing or changed")
        candidates = []
        for receipt in attempt.get("checkpoint_history", []):
            if receipt.get("pruned_at") is not None:
                continue
            original = Path(receipt["path"])
            original = original if original.is_absolute() else self.guard.root / original
            target = self.guard.path(original)
            if (receipt.get("attempt_id") != self.attempt_id or original.is_symlink()
                    or target.parent != directory
                    or re.fullmatch(r"step-[0-9]{9,}-[a-f0-9]{8}\.pt", target.name) is None):
                raise RuntimeSafetyError("checkpoint history contains an unsafe retention path")
            if str(target) in pins:
                continue
            if not target.is_file() or sha256_file(target) != receipt["sha256"]:
                raise RuntimeSafetyError("unprotected checkpoint changed or disappeared; refuse retention")
            candidates.append((receipt, target, target.stat().st_size))
        if not candidates:
            return {"removed": 0, "bytes": 0, "kept_pins": len(pins)}
        authorized_at = _now()
        for receipt, _, _ in candidates:
            receipt["prune_authorized_at"] = authorized_at
        self._flush()  # No unlink occurs unless the complete removal plan commits.
        removed_bytes = 0
        for receipt, target, size in candidates:
            # Recheck exact identity at unlink time, not a wildcard or directory.
            if target.is_symlink() or self.guard.path(target) != target or sha256_file(target) != receipt["sha256"]:
                raise RuntimeSafetyError("checkpoint changed after retention was authorized")
            target.unlink()
            receipt["pruned_at"] = _now()
            removed_bytes += size
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._flush()
        return {"removed": len(candidates), "bytes": removed_bytes, "kept_pins": len(pins)}

    def latest_checkpoint(self) -> Path | None:
        self.assert_owned()
        receipt = self.registry["latest_checkpoint"]
        if receipt is None:
            return None
        path = self.guard.path(receipt["path"])
        if not path.is_relative_to(self.run_dir) or sha256_file(path) != receipt["sha256"]:
            raise RuntimeSafetyError("latest checkpoint path or SHA-256 differs")
        return path

    def update_progress(self, *, epoch: int, global_step: int, next_batch: int,
                        validation_count: int, loss: float | None = None,
                        metrics: dict | None = None) -> None:
        """Publish bounded current progress; call periodically, not every tensor op."""
        self.assert_owned()
        attempt = self.registry["attempts"][-1]
        if attempt["status"] != "running":
            raise RuntimeSafetyError("cannot update a finished attempt")
        value = {"epoch": epoch, "global_step": global_step, "next_batch": next_batch,
                 "validation_count": validation_count, "checked_at": _now()}
        for key in ("epoch", "global_step", "next_batch", "validation_count"):
            if type(value[key]) is not int or value[key] < 0:
                raise RuntimeSafetyError(f"invalid progress {key}")
        previous = attempt.get("progress", {})
        for key in ("epoch", "global_step", "validation_count"):
            if value[key] < previous.get(key, 0):
                raise RuntimeSafetyError(f"progress {key} moved backwards within an attempt")
        if loss is not None:
            value["loss"] = float(loss)
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise TypeError("metrics must be a dictionary")
            value["metrics"] = metrics
        assert_finite(value, "progress")
        # Enforce JSON-only monitoring fields before changing in-memory state.
        value = json.loads(json.dumps(value, allow_nan=False))
        attempt["progress"] = value
        self._flush()

    def record_event(self, kind: str, details: dict | None = None) -> Path:
        """Write a small immutable event, intended for phase/failure boundaries."""
        self.assert_owned()
        attempt = self.registry["attempts"][-1]
        if attempt["status"] != "running":
            raise RuntimeSafetyError("cannot append an event to a finished attempt")
        if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", kind):
            raise ValueError("invalid event kind")
        if details is not None and not isinstance(details, dict):
            raise TypeError("event details must be a JSON dictionary")
        value = {"kind": kind, "at": _now(), "attempt_id": self.attempt_id,
                 "owner": self.owner, "details": details or {}}
        assert_finite(value, "event")
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > 65536:
            raise ValueError("event exceeds 64 KiB; store large diagnostics separately")
        path = self.attempt_dir / "events" / f"{uuid.uuid4().hex}-{kind}.json"
        _atomic_write(self.guard, path, lambda stream: stream.write(raw), replace=False)
        attempt["event_count"] += 1
        attempt["last_event"] = {"kind": kind, "path": str(path.relative_to(self.guard.root))}
        self._flush()
        return path

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.assert_owned()
        if status not in {"completed", "stopped", "failed"}:
            raise ValueError("invalid final attempt status")
        attempt = self.registry["attempts"][-1]
        if attempt["status"] != "running":
            raise RuntimeSafetyError("attempt is already finished")
        attempt.update(status=status, ended_at=_now())
        if error is not None:
            attempt["error"] = error[:2000]
        self._flush()

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.registry["attempts"][-1]["status"] == "running":
                self.finish("failed" if exc is not None else "stopped",
                            f"{exc_type.__name__}: {exc}" if exc is not None else None)
        finally:
            self._release()
        return False
