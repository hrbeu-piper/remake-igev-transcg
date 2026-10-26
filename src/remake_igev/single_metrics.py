"""Isolated reference-only grouping around the unchanged exact metric kernel.

The kernel is loaded into a distinct registered module because its accumulator
uses module-level grouping constants. No global of remake_igev.metrics changes.
All arithmetic, morphology and eight-pass float64 radix selection are reused.
The internal support-known field stays False solely for kernel record checks;
public records and the new protocol expose no stereo/support strata.
"""
from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
import re
import sys

from .runtime import canonical_hash, sha256_file

_name = __package__ + '._single_metrics_kernel'
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name('metrics.py'))
    _kernel = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _kernel
    try:
        _spec.loader.exec_module(_kernel)
    except BaseException:
        del sys.modules[_name]
        raise
else:
    _kernel = sys.modules[_name]

REGIONS = _kernel.REGIONS
STRATA = ('all', 'raw_nonzero', 'raw_hole', 'D435', 'L515')
GROUPS = tuple(f'{region}/{stratum}' for region in REGIONS for stratum in STRATA)
PROTOCOL = copy.deepcopy(_kernel.PROTOCOL)
PROTOCOL.update(version='transcg_reference_only_metrics_v1', strata=list(STRATA),
                data_protocol='reference_only_v1', kernel_source_sha256=sha256_file(Path(_kernel.__file__)))
PROTOCOL.pop('geometry_support', None)
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)


def _metadata(sample_id, scene_id, camera_id, mode, support_known):
    _kernel._require(type(scene_id) is int and 1 <= scene_id <= 130 and type(camera_id) is int
                     and camera_id in (1, 2), 'Invalid reference scene/camera')
    match = re.fullmatch(r'scene([0-9]{3})/view([0-9]{3})/camera([12])', sample_id or '')
    _kernel._require(match is not None and int(match[1]) == scene_id and int(match[3]) == camera_id
                     and 0 <= int(match[2]) < 240, 'Reference ID differs')
    _kernel._require(mode == 'reference_only' and support_known is False,
                     'Single-view metrics cannot claim stereo or geometric support')
    return {'sample_id': sample_id, 'scene_id': scene_id, 'view_id': int(match[2]),
            'camera_id': camera_id, 'mode': 'reference_only', 'geometry_support_known': False}


_kernel.STRATA, _kernel.GROUPS = STRATA, GROUPS
_kernel.PROTOCOL, _kernel.PROTOCOL_SHA256 = PROTOCOL, PROTOCOL_SHA256
_kernel._metadata = _metadata
MetricAccumulator = _kernel.MetricAccumulator
MetricError = _kernel.MetricError
MAX_RECORD_BYTES, MAX_JSON_BYTES = _kernel.MAX_RECORD_BYTES, _kernel.MAX_JSON_BYTES


def evaluate_sample(pred_m, gt_m, transparent_mask, raw_m, *, sample_id, scene_id, camera_id):
    sample = _kernel.evaluate_sample(pred_m, gt_m, transparent_mask, raw_m, sample_id=sample_id,
                                    scene_id=scene_id, camera_id=camera_id, mode='reference_only')
    # The dataclass constructor's default was compiled before kernel binding.
    return replace(sample, protocol_sha256=PROTOCOL_SHA256)


def public_record(sample):
    result = sample.to_record()
    result['kind'] = 'single_view_sample_depth_metrics'
    result['metadata'].pop('geometry_support_known')
    return result
