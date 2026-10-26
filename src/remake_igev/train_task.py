"""Model-specific training/validation adapters; one objective per reference image.

Per-reference region normalization makes the objective independent of physical
microbatch partitioning. Evaluation MAE remains pixel weighted. Both choices are
explicit protocol fields and must be shared by the baseline and fusion groups.
"""
from dataclasses import fields, is_dataclass
import torch

from .objectives import LossConfig, compute_objective
from .runtime import assert_finite


def select_reference(value, index: int):
    """Slice all batch-leading tensors, including this pass's candidate grids."""
    if isinstance(value, torch.Tensor):
        return value[index:index + 1]
    if is_dataclass(value):
        return type(value)(**{field.name: select_reference(getattr(value, field.name), index)
                              for field in fields(value)})
    if isinstance(value, dict):
        return {key: select_reference(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(select_reference(item, index) for item in value)
    if value is None:
        return None
    raise TypeError(f'Unsupported batched objective value: {type(value)}')


class ReferenceObjective:
    def __init__(self, config: LossConfig, *, iterations: int):
        self.config, self.iterations = config, iterations
        self.last_diagnostics = {}

    def __call__(self, model, batch):
        output = model(batch['inputs'], iterations=self.iterations)
        batch_size = output.depth.shape[0]
        losses, counts = [], {}
        for index in range(batch_size):
            prediction = select_reference(output, index)
            result = compute_objective(
                prediction.geometry if prediction.geometry is not None else prediction.depth,
                select_reference(batch['targets'], index), batch['inputs']['K_ref'][index:index + 1],
                baseline_depth=prediction.baseline_depth, K_candidates=prediction.K_candidates,
                config=self.config,
            )
            losses.append(result.loss)
            for key, value in result.pixel_counts.items():
                counts[key] = counts.get(key, 0) + value
        self.last_diagnostics = {
            'supervision_pixel_counts': counts,
            'baseline_positive_fraction': float((output.baseline_depth.detach() > 0).float().mean()),
            'geometry_valid_pixels': int(output.geometry.valid.sum()) if output.geometry is not None else 0,
        }
        return torch.stack(losses)


def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_tensors(item, device) for item in value)
    return value


@torch.no_grad()
def validate(model, dataset, *, device, iterations: int, on_progress=None):
    """Complete deterministic validation; no test input and no dropped samples.

    This is a checkpoint-selection metric, not the complete final test report.
    A failed validation restarts from its beginning after restoring the saved
    pre-validation checkpoint; training updates are not repeated.
    """
    if getattr(dataset, 'split', None) != 'val':
        raise ValueError('Checkpoint selection requires the isolated val split')
    from torch.utils.data import default_collate
    previous = model.training
    model.eval()
    totals = {'valid_abs_error_sum_m': 0.0, 'valid_pixels': 0,
              'transparent_abs_error_sum_m': 0.0, 'transparent_pixels': 0, 'samples': 0}
    try:
        for index in range(len(dataset)):
            batch = move_tensors(default_collate([dataset[index]]), device)
            output = model(batch['inputs'], iterations=iterations).depth
            assert_finite(output, 'validation_prediction')
            gt = batch['targets']['depth_gt']
            valid = batch['targets']['gt_valid'] & (gt > 0)
            transparent = valid & batch['targets']['transparent_mask']
            errors = (output - gt).abs().double()
            totals['valid_abs_error_sum_m'] += float(errors[valid].sum())
            totals['transparent_abs_error_sum_m'] += float(errors[transparent].sum())
            totals['valid_pixels'] += int(valid.sum())
            totals['transparent_pixels'] += int(transparent.sum())
            totals['samples'] += 1
            if on_progress is not None and ((index + 1) % 25 == 0 or index + 1 == len(dataset)):
                on_progress(index + 1)
        if not totals['valid_pixels'] or not totals['transparent_pixels']:
            raise ValueError('Validation has no valid overall or transparent GT pixels')
        totals['valid_mae_m'] = totals['valid_abs_error_sum_m'] / totals['valid_pixels']
        totals['transparent_mae_m'] = totals['transparent_abs_error_sum_m'] / totals['transparent_pixels']
        assert_finite(totals, 'validation_metrics')
        return totals
    finally:
        model.train(previous)
