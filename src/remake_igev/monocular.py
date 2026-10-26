"""Frozen Depth Anything V2 Small RGB priors and immutable, guarded caches.

ReMake's inspected runner sends the raw DA forward result to remake_trainer,
which only resizes it with bilinear/align_corners=False. There is no min-max or
metric normalization. This adapter preserves that output convention but fixes
RGB preprocessing: original RGB -> vendor Resize/NormalizeImage/PrepareForNet,
without image2tensor's BGR conversion or automatic device selection. It is a
new, recorded preprocessing protocol, not bitwise legacy reproduction.

Production cache generation requires explicit CUDA. CPU is available only with
allow_cpu_checks=True and produces a distinct cpu_interface_check fingerprint;
ordinary readers reject these check caches. DataLoader providers are read-only
and contain no model: a miss or corruption raises, never runs CUDA in a worker.

The optional training-grid cache keeps the same two interpolation operations:
DA output -> native RGB on the inference device, then native float32 -> final
grid on CPU through dataset.SpatialTransform. Only the second result is stored;
no quantization or direct DA-output-to-training-grid shortcut is used.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping, Any
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

try:
    from .runtime import ResourceGuard, canonical_hash, sha256_file, require_cuda_device
    from . import dataset as _dataset
except ImportError:
    from runtime import ResourceGuard, canonical_hash, sha256_file, require_cuda_device
    import dataset as _dataset


MODEL_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
PREPROCESS_VERSION = "rgb-native-lower-bound14-bicubic-imagenet-raw-relative-align-false-v1"
MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
CACHE_GRID_VERSION = "native-then-dataset-grid-float32-v1"
# Each immutable archive is checked against this bound before publication.
# Capacity planning therefore never assumes a favourable compression ratio.
CACHE_ENTRY_OVERHEAD_BYTES = 64 * 1024


class MonocularError(RuntimeError):
    pass


class CacheMissError(MonocularError):
    pass


class CacheCorruptionError(MonocularError):
    pass


class CacheBusyError(MonocularError):
    pass


def _not_a_worker() -> None:
    if torch.utils.data.get_worker_info() is not None:
        raise MonocularError("Prior generation is forbidden inside DataLoader workers; precompute the cache")


def _rgb(rgb: np.ndarray) -> np.ndarray:
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise MonocularError("Expected original RGB uint8 HxWx3 image")
    return np.ascontiguousarray(rgb)


def rgb_pixel_sha256(rgb: np.ndarray) -> str:
    rgb = _rgb(rgb)
    header = f"RGB:uint8:{rgb.shape[0]}:{rgb.shape[1]}\n".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def read_original_rgb(path: Path | str) -> tuple[np.ndarray, str]:
    """Decode and hash the same encoded image bytes, avoiding a read/hash race."""
    try:
        encoded = Path(path).read_bytes()
        with Image.open(io.BytesIO(encoded)) as image:
            if image.mode != "RGB":
                raise MonocularError(f"Expected RGB image, found {image.mode}: {path}")
            rgb = np.array(image)
    except (OSError, ValueError) as exc:
        raise MonocularError(f"Cannot decode original RGB {path}: {exc}") from exc
    return _rgb(rgb), hashlib.sha256(encoded).hexdigest()


def _vendor_sources(vendor_src: Path) -> dict:
    package = vendor_src / "depth_anything_v2"
    if not (package / "dpt.py").is_file() or not (package / "util/transform.py").is_file():
        raise MonocularError("vendor_src must contain depth_anything_v2/dpt.py and util/transform.py")
    files = {}
    for path in sorted(package.rglob("*.py")):
        if not path.resolve().is_relative_to(vendor_src):
            raise MonocularError("Vendor source symlink escapes vendor_src")
        files[path.relative_to(vendor_src).as_posix()] = sha256_file(path)
    return {"files": files, "sha256": canonical_hash(files)}


def _import_vendor(vendor_src: Path):
    existing = sys.modules.get("depth_anything_v2")
    if existing is not None:
        locations = list(getattr(existing, "__path__", []))
        if not locations or any(not Path(path).resolve().is_relative_to(vendor_src) for path in locations):
            raise MonocularError("A different depth_anything_v2 package is already imported")
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))
    try:
        dpt = importlib.import_module("depth_anything_v2.dpt")
        transform = importlib.import_module("depth_anything_v2.util.transform")
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise MonocularError(f"Cannot import the pinned vendor dependencies: {exc}") from exc
    for module in (dpt, transform):
        if not Path(module.__file__).resolve().is_relative_to(vendor_src):
            raise MonocularError("Imported module does not belong to the declared vendor_src")
    return dpt, transform, cv2


class FrozenDepthAnythingV2Small:
    """Strict fixed-weight inference with explicit device and RGB-only input."""

    def __init__(self, vendor_src: str | Path, weights: str | Path, *,
                 device: str | torch.device, input_size: int = 518,
                 allow_cpu_checks: bool = False):
        _not_a_worker()
        if type(input_size) is not int or input_size < 28 or input_size % 14:
            raise MonocularError("input_size must be a multiple of 14 and at least 28")
        requested = torch.device(device)
        if requested.type == "cpu":
            if not allow_cpu_checks:
                raise MonocularError("CPU prior inference requires explicit allow_cpu_checks=True")
            self.device = requested
            execution_mode = "cpu_interface_check"
        elif requested.type == "cuda":
            self.device = require_cuda_device(requested)
            execution_mode = "cuda_cache_generation"
        else:
            raise MonocularError("Only explicit CUDA or opted-in CPU interface checks are supported")
        self.vendor_src = Path(vendor_src).resolve(strict=True)
        self.weights = Path(weights).resolve(strict=True)
        actual_hash = sha256_file(self.weights)
        if actual_hash != MODEL_SHA256:
            raise MonocularError(f"DA V2 Small weight SHA-256 mismatch: {actual_hash}")
        self.source_receipt = _vendor_sources(self.vendor_src)
        dpt, transform, cv2 = _import_vendor(self.vendor_src)
        self.transforms = (
            transform.Resize(width=input_size, height=input_size, resize_target=False,
                             keep_aspect_ratio=True, ensure_multiple_of=14,
                             resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
            transform.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transform.PrepareForNet(),
        )
        self.input_size = input_size
        self.model = dpt.DepthAnythingV2(**MODEL_CONFIG)
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise MonocularError("Expected the fixed plain DA V2 Small tensor state_dict")
        if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
            raise MonocularError("Nonfinite value in DA weights")
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.fingerprint = {
            "schema_version": 1, "model": "DepthAnythingV2-Small", "model_config": copy.deepcopy(MODEL_CONFIG),
            "model_sha256": actual_hash, "vendor_source_sha256": self.source_receipt["sha256"],
            "adapter_source_sha256": sha256_file(Path(__file__)),
            "preprocess_version": PREPROCESS_VERSION, "input_size": input_size,
            "input_color": "RGB_uint8", "output_normalization": "none_raw_DA_relative",
            "output_resize": "bilinear_align_corners_false_to_original_rgb",
            "execution_mode": execution_mode, "inference_dtype": "float32",
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "opencv_version": str(cv2.__version__),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
        if self.device.type == "cuda":
            self.fingerprint["cuda_capability"] = list(torch.cuda.get_device_capability(self.device))

    def preprocess_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        sample = {"image": _rgb(rgb).astype(np.float32) / 255.0}
        for transform in self.transforms:
            sample = transform(sample)
        return torch.from_numpy(sample["image"]).unsqueeze(0)

    @torch.inference_mode()
    def predict_rgb(self, rgb: np.ndarray) -> np.ndarray:
        _not_a_worker()
        rgb = _rgb(rgb)
        if (bool(torch.backends.cuda.matmul.allow_tf32) != self.fingerprint["cuda_matmul_allow_tf32"]
                or bool(torch.backends.cudnn.allow_tf32) != self.fingerprint["cudnn_allow_tf32"]):
            raise MonocularError("Precision settings changed after fingerprint creation; recreate the predictor")
        self.model.eval()
        tensor = self.preprocess_rgb(rgb).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.model(tensor)
            if output.ndim != 3 or output.shape[0] != 1:
                raise MonocularError("Unexpected DA forward output shape")
            output = F.interpolate(output[:, None], size=rgb.shape[:2], mode="bilinear",
                                   align_corners=False)[0, 0]
        array = output.float().cpu().numpy().copy()
        _valid_depth(array, rgb.shape[:2])
        return array


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise MonocularError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _output_hw(value: Any) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)):
        raise MonocularError("Cache output_hw must contain two positive integer dimensions")
    return tuple(value)


def _grid_storage(output_hw: tuple[int, int]) -> dict:
    return {"version": CACHE_GRID_VERSION, "output_hw": list(_output_hw(output_hw)),
            "dtype": "float32", "encoding": "npz_stored",
            "resize_sequence": "predict_rgb_to_native_then_dataset_spatial_transform_on_cpu",
            "resize_version": _dataset.RELATIVE_GRID_VERSION,
            "spatial_transform_source_sha256": sha256_file(Path(_dataset.__file__)),
            "supported_transform": "resize_only_no_native_crop"}


def grid_cache_fingerprint(predictor_fingerprint: Mapping,
                           output_hw: tuple[int, int] = (480, 640)) -> dict:
    """Bind storage geometry without changing the predictor's native output.

    Keep the predictor fingerprint itself native. Pass the returned fingerprint
    both to RelativeDepthCache and to the training fingerprint JSON. Readers
    and producers then agree on the final grid without trainer-specific code.
    """
    if not isinstance(predictor_fingerprint, Mapping) or "cache_storage" in predictor_fingerprint:
        raise MonocularError("Expected a native predictor fingerprint without cache_storage")
    result = copy.deepcopy(dict(predictor_fingerprint))
    result["cache_storage"] = _grid_storage(output_hw)
    return result


def cache_storage_budget(reference_count: int, output_hw: tuple[int, int]) -> dict:
    """An uncompressed archive bound, plus a separate filesystem allowance.

    This is a full-cache planning bound, not an assertion about free capacity or
    simultaneous writers. ResourceGuard and per-publication checks remain in
    force. Temporary publication holds at most one extra archive per writer.
    """
    if type(reference_count) is not int or reference_count < 0:
        raise MonocularError("reference_count must be a nonnegative integer")
    height, width = _output_hw(output_hw)
    payload = height * width * np.dtype(np.float32).itemsize
    archive = payload + CACHE_ENTRY_OVERHEAD_BYTES
    filesystem_allowance = (reference_count + 512) * 4096
    return {"references": reference_count, "output_hw": [height, width],
            "array_bytes": reference_count * payload,
            "maximum_archive_bytes_each": archive,
            "maximum_archives_bytes": reference_count * archive,
            "filesystem_allowance_bytes": filesystem_allowance,
            "maximum_temporary_bytes_one_writer": archive,
            "planned_bytes_with_one_writer": reference_count * archive + filesystem_allowance + archive,
            "compression_assumed": False}


def _fingerprint(value: Mapping, *, allow_cpu_checks: bool) -> dict:
    if not isinstance(value, Mapping):
        raise MonocularError("Cache requires an explicit predictor fingerprint")
    result = copy.deepcopy(dict(value))
    for name in ("model_sha256", "vendor_source_sha256", "adapter_source_sha256"):
        _sha(result.get(name), name)
    if (result.get("schema_version") != 1 or result.get("model_sha256") != MODEL_SHA256
            or result.get("preprocess_version") != PREPROCESS_VERSION
            or result.get("output_normalization") != "none_raw_DA_relative"):
        raise MonocularError("Cache fingerprint does not describe this fixed DA RGB protocol")
    size = result.get("input_size")
    if type(size) is not int or size < 28 or size % 14:
        raise MonocularError("Invalid cache input_size")
    mode = result.get("execution_mode")
    if mode == "cpu_interface_check":
        if not allow_cpu_checks:
            raise MonocularError("CPU interface-check cache cannot be used as a production prior")
    elif mode != "cuda_cache_generation":
        raise MonocularError("Unsupported cache execution mode")
    if "cache_storage" in result:
        storage = result["cache_storage"]
        if (not isinstance(storage, Mapping)
                or dict(storage) != _grid_storage(storage.get("output_hw"))):
            raise MonocularError("Cache storage fingerprint has an incompatible grid, version or source")
    return result


def _valid_depth(depth: np.ndarray, shape: tuple[int, int]) -> None:
    if (not isinstance(depth, np.ndarray) or depth.dtype != np.float32
            or depth.shape != tuple(shape) or not np.isfinite(depth).all() or (depth < 0).any()):
        raise MonocularError("Expected finite nonnegative float32 relative depth at the declared resolution")


@dataclass(frozen=True)
class CacheResult:
    depth: np.ndarray
    key: str
    path: Path
    hit: bool


class RelativeDepthCache:
    """Immutable NPZ entries with encoded-image, RGB-pixel, model and code hashes."""

    def __init__(self, guard: ResourceGuard, cache_dir: Path | str, *,
                 fingerprint: Mapping, allow_cpu_checks: bool = False):
        if not isinstance(guard, ResourceGuard):
            raise MonocularError("A ResourceGuard is required for the prior cache")
        self.guard = guard
        self.cache_dir = guard.path(cache_dir)
        self.fingerprint = _fingerprint(fingerprint, allow_cpu_checks=allow_cpu_checks)
        self.allow_cpu_checks = allow_cpu_checks
        self.fingerprint_sha256 = canonical_hash(self.fingerprint)
        storage = self.fingerprint.get("cache_storage")
        self.output_hw = None if storage is None else tuple(storage["output_hw"])

    def identity(self, rgb: np.ndarray, image_file_sha256: str) -> dict:
        rgb = _rgb(rgb)
        return {"schema_version": 2,
                "image_file_sha256": _sha(image_file_sha256, "original encoded image SHA-256"),
                "original_rgb_sha256": rgb_pixel_sha256(rgb), "original_shape": list(rgb.shape[:2]),
                "stored_shape": list(self.output_hw or rgb.shape[:2]),
                "predictor": copy.deepcopy(self.fingerprint)}

    def entry_path(self, key: str) -> Path:
        _sha(key, "cache key")
        return self.guard.path(self.cache_dir / self.fingerprint["execution_mode"] /
                               self.fingerprint_sha256 / key[:2] / f"{key}.npz")

    def _read(self, identity: dict) -> CacheResult:
        key = canonical_hash(identity)
        path = self.entry_path(key)
        if not path.is_file():
            if path.exists():
                raise CacheCorruptionError(f"Cache entry is not a file: {path}")
            raise CacheMissError(f"Relative-depth cache miss: {key}; precompute before DataLoader use")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"depth", "receipt_utf8"}:
                    raise CacheCorruptionError("Unexpected cache archive fields")
                depth = np.array(archive["depth"], copy=True)
                encoded = archive["receipt_utf8"]
                if encoded.dtype != np.uint8 or encoded.ndim != 1:
                    raise CacheCorruptionError("Invalid cache receipt encoding")
                receipt = json.loads(encoded.tobytes().decode("utf-8"))
            if not isinstance(receipt, dict) or receipt.get("identity") != identity or receipt.get("key") != key:
                raise CacheCorruptionError("Cache identity mismatch")
            _valid_depth(depth, tuple(identity["stored_shape"]))
            if hashlib.sha256(depth.tobytes(order="C")).hexdigest() != receipt.get("depth_sha256"):
                raise CacheCorruptionError("Cached relative-depth checksum mismatch")
        except (OSError, ValueError, KeyError, UnicodeError, EOFError, zipfile.BadZipFile, MonocularError) as exc:
            if isinstance(exc, CacheMissError):
                raise
            raise CacheCorruptionError(f"Corrupt prior cache {path}: {exc}") from exc
        return CacheResult(depth, key, path, True)

    def read(self, rgb: np.ndarray, image_file_sha256: str) -> CacheResult:
        return self._read(self.identity(rgb, image_file_sha256))

    @contextmanager
    def _writer_lock(self, path: Path):
        self.guard.mkdir(path.parent)
        lock_path = self.guard.path(path.with_suffix(".lock"))
        with lock_path.open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CacheBusyError(f"Another process is generating cache entry {path.name}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        # Keep the lock inode: unlinking it permits concurrent users of different
        # inodes to believe they both hold the same per-key lock.

    def _publish(self, path: Path, depth: np.ndarray, identity: dict) -> None:
        path = self.guard.path(path)
        _valid_depth(depth, tuple(identity["stored_shape"]))
        maximum_bytes = depth.nbytes + CACHE_ENTRY_OVERHEAD_BYTES
        if shutil.disk_usage(self.guard.root).free < self.guard.minimum_free_bytes + maximum_bytes:
            raise MonocularError("Insufficient space to publish a prior while preserving the resource reserve")
        receipt = {"key": canonical_hash(identity), "identity": identity,
                   "depth_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest()}
        encoded = np.frombuffer(json.dumps(receipt, sort_keys=True, allow_nan=False,
                                          separators=(",", ":")).encode("utf-8"), dtype=np.uint8)
        fd, temporary_name = tempfile.mkstemp(prefix=".prior-", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, depth=depth, receipt_utf8=encoded)
                if stream.tell() > maximum_bytes:
                    raise MonocularError("Cache archive exceeds its uncompressed capacity bound")
                stream.flush()
                os.fsync(stream.fileno())
            self.guard.path(path)
            # Exclusive link publication cannot overwrite an existing cache.
            os.link(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def populate_file(self, image_path: str | Path,
                      predictor: FrozenDepthAnythingV2Small) -> CacheResult:
        _not_a_worker()
        if not isinstance(predictor, FrozenDepthAnythingV2Small):
            raise MonocularError("Cache generation requires the strict fixed-weight DA predictor")
        expected = (predictor.fingerprint if self.output_hw is None else
                    grid_cache_fingerprint(predictor.fingerprint, self.output_hw))
        if expected != self.fingerprint:
            raise MonocularError("Cache and predictor fingerprints differ")
        rgb, image_sha256 = read_original_rgb(image_path)
        identity = self.identity(rgb, image_sha256)
        try:
            return self._read(identity)
        except CacheMissError:
            pass
        key = canonical_hash(identity)
        path = self.entry_path(key)
        with self._writer_lock(path):
            try:
                return self._read(identity)
            except CacheMissError:
                pass
            depth = predictor.predict_rgb(rgb)
            _valid_depth(depth, rgb.shape[:2])
            if self.output_hw is not None:
                # predict_rgb has already restored native RGB resolution on its
                # original device and copied float32 to CPU. Reuse the exact
                # dataset operation, including its no-resize clone branch.
                tensor = torch.from_numpy(depth.copy()).unsqueeze(0)
                depth = _dataset.SpatialTransform(output_size=self.output_hw).apply(
                    tensor, nearest=False)[0].numpy().copy()
                _valid_depth(depth, self.output_hw)
            self._publish(path, depth, identity)
        checked = self._read(identity)
        return CacheResult(checked.depth, key, path, False)

    def provider(self, image_paths: Mapping[str, str | Path]) -> "CachedRelativeDepthProvider":
        return CachedRelativeDepthProvider(self, image_paths)


class CachedRelativeDepthProvider:
    """Dataset callback: hash/read only; never holds a model or generates a miss.

    image_paths maps each fixed manifest sample_id to its original RGB file.
    File bytes are rehashed per read, and the caller's decoded RGB is also bound
    to the cache key. Changing image contents, source, input_size or weights
    therefore invalidates the entry instead of silently reusing stale priors.
    """

    def __init__(self, cache: RelativeDepthCache, image_paths: Mapping[str, str | Path]):
        self.cache = cache
        self.image_paths = {str(key): str(Path(value).absolute()) for key, value in image_paths.items()}

    def __call__(self, sample_id: str, rgb: np.ndarray) -> np.ndarray | _dataset.ResizedRelativeDepth:
        if sample_id not in self.image_paths:
            raise CacheMissError(f"No original RGB path registered for sample {sample_id}")
        path = Path(self.image_paths[sample_id])
        try:
            original_sha = sha256_file(path)
        except OSError as exc:
            raise CacheMissError(f"Missing original RGB for cached sample {sample_id}: {path}") from exc
        depth = self.cache.read(rgb, original_sha).depth
        if self.cache.output_hw is None:
            return depth
        return _dataset.ResizedRelativeDepth(depth, tuple(rgb.shape[:2]), self.cache.output_hw)
