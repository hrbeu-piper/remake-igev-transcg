"""Deterministic, sample-weighted FP32 training at complete step boundaries.

Protocol v1 uses synchronous, deterministic dataset indexing and no stochastic
augmentation or DataLoader workers. ``start_sample`` counts consumed positions
in an epoch permutation, not dataset IDs or physical batches. Only a completed
effective-batch boundary (or the epoch end) is resumable. Persist its cursor and
the complete runtime checkpoint, including global RNG, before resuming.

``loss_fn(model, collated_microbatch)`` must return float32 ``[B]`` losses: each
reference sample's objective is normalized on that sample alone. A batch-wide
masked mean repeated B times violates this contract. For ReMake, split the
batched output/targets and call compute_objective separately for each sample.
The engine sums these losses divided by the actual effective-group size,
including the final short group. This changes the explicit loss normalization
protocol from a pooled pixel mean to an equal mean over reference samples.

Physical-batch changes preserve this objective for batch-separable models with
fixed normalization statistics, up to floating-point reduction differences.
Dropout/RNG draws can change with physical batch shape; exact resume therefore
requires the same physical batch when stochastic forward operations are used.
AMP, random augmentation, worker prefetch, DDP and gradient clipping are absent.
The callback owns stage/loss selection, scheduler, provenance and checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import default_collate

from .runtime import RuntimeSafetyError, assert_finite, require_cuda_device


SAMPLER_VERSION = "sha256-seed-epoch-randperm-v1"
LOSS_NORMALIZATION = "mean-of-per-reference-objectives-v1"


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def epoch_order(dataset_size: int, *, epoch: int, seed: int) -> tuple[int, ...]:
    """CPU private-generator permutation; never consumes the model's RNG."""
    _integer(dataset_size, "dataset_size")
    _integer(epoch, "epoch")
    _integer(seed, "seed")
    digest = hashlib.sha256(f"{SAMPLER_VERSION}:{seed}:{epoch}".encode("ascii")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    return tuple(torch.randperm(dataset_size, generator=generator).tolist())


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        if value.is_complex():
            raise ValueError("complex batch tensors are not supported")
        return value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


@dataclass(frozen=True)
class StepState:
    epoch: int
    seed: int
    dataset_size: int
    global_step: int
    next_sample_index: int
    next_batch: int
    sample_indices: tuple[int, ...]
    samples_in_step: int
    loss_mean: float
    nonzero_gradient_parameters: int

    def progress(self, best_metrics: dict | None = None) -> dict:
        """Compatible with runtime.save_checkpoint; next_batch is logical."""
        return {"epoch": self.epoch, "global_step": self.global_step,
                "next_batch": self.next_batch, "accumulation_position": 0,
                "best_metrics": dict(best_metrics or {})}

    def data_state(self) -> dict:
        return {"sampler_epoch": self.epoch, "sampler_seed": self.seed,
                "next_sample_index": self.next_sample_index, "dataset_size": self.dataset_size}


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    next_sample_index: int
    global_step: int
    optimizer_steps: int
    consumed_samples: int
    loss_mean: float | None
    completed: bool


def _validate_model_optimizer(model: nn.Module, optimizer, device: torch.device) -> None:
    parameters = dict(model.named_parameters())
    trainable = {id(p) for p in parameters.values() if p.requires_grad}
    if not trainable:
        raise RuntimeSafetyError("no trainable model parameters")
    owned = {id(p) for p in parameters.values()}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if (len(set(optimizer_ids)) != len(optimizer_ids)
            or not trainable.issubset(optimizer_ids) or not set(optimizer_ids).issubset(owned)):
        raise RuntimeSafetyError("optimizer must own every trainable parameter exactly once and no external parameters")
    for name, value in list(parameters.items()) + list(model.named_buffers()):
        if value.device != device:
            raise RuntimeSafetyError(f"{name}: model tensor is not on the requested device")
        if value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
            raise RuntimeSafetyError(f"{name}: this engine requires float32 model tensors")
    if any(p.grad is not None for p in parameters.values()):
        raise RuntimeSafetyError("pending gradients at epoch entry; resume only after step and zero_grad")
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
           for module in model.modules()):
        raise RuntimeSafetyError("training BatchNorm statistics depend on physical batch; freeze them for protocol v1")
    assert_finite(model.state_dict(), "initial model")
    assert_finite(optimizer.state_dict(), "initial optimizer")


def _gradient_check(model: nn.Module) -> int:
    nonzero = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not parameter.requires_grad:
            raise RuntimeSafetyError(f"unexpected gradient on frozen parameter {name}")
        if gradient.is_sparse:
            raise RuntimeSafetyError("sparse gradients are not supported by protocol v1")
        assert_finite(gradient, f"gradient.{name}")
        nonzero += int(bool(torch.count_nonzero(gradient)))
    if nonzero == 0:
        raise RuntimeSafetyError("all trainable gradients are absent or zero; refuse optimizer step")
    return nonzero


def run_epoch(
    model: nn.Module, optimizer, dataset, loss_fn: Callable[[nn.Module, Any], Tensor], *,
    epoch: int, seed: int, start_sample: int = 0, global_step: int = 0,
    effective_batch: int, physical_batch: int, device: str | torch.device,
    on_step: Callable[[StepState], bool | None] | None = None,
    max_steps: int | None = None, collate_fn: Callable = default_collate,
    allow_cpu_checks: bool = False,
) -> EpochResult:
    """Run/resume one deterministic epoch, without saving or changing stages.

    on_step runs after optimizer.step, finite-state checks and zero_grad. It may
    update a scheduler, checkpoint using StepState, and return False to stop at
    that safe boundary. Returning None/True continues. A callback exception is
    propagated: the current step has happened, so reload the last good complete
    checkpoint rather than retrying it in place. Forward/backward failures clear
    pending gradients and do not call optimizer.step. An optimizer/post-step
    failure can mutate state and likewise requires reloading the last checkpoint.

    CPU is permitted only by explicit diagnostic opt-in, never automatic fallback.
    The caller moves model/optimizer to device before invoking this function.
    """
    _integer(epoch, "epoch")
    _integer(seed, "seed")
    _integer(start_sample, "start_sample")
    _integer(global_step, "global_step")
    _integer(effective_batch, "effective_batch", 1)
    _integer(physical_batch, "physical_batch", 1)
    if max_steps is not None:
        _integer(max_steps, "max_steps")
    size = _integer(len(dataset), "dataset_size")
    if start_sample > size or (start_sample != size and start_sample % effective_batch):
        raise ValueError("start_sample must be a completed effective-batch boundary or dataset_size")
    if not callable(loss_fn) or not callable(collate_fn) or (on_step is not None and not callable(on_step)):
        raise TypeError("loss_fn, collate_fn and on_step must be callable")
    device = torch.device(device)
    if device.type == "cpu":
        if not allow_cpu_checks:
            raise RuntimeSafetyError("CPU requires explicit allow_cpu_checks; never use as training fallback")
    else:
        device = require_cuda_device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    model.train()
    _validate_model_optimizer(model, optimizer, device)
    order = epoch_order(size, epoch=epoch, seed=seed)
    cursor, steps, consumed, total_loss = start_sample, 0, 0, 0.0
    while cursor < size and (max_steps is None or steps < max_steps):
        end = min(cursor + effective_batch, size)
        indices = order[cursor:end]
        actual_size = len(indices)
        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        try:
            for offset in range(0, actual_size, physical_batch):
                micro_indices = indices[offset:offset + physical_batch]
                batch = _to_device(collate_fn([dataset[index] for index in micro_indices]), device)
                assert_finite(batch, "training batch")
                with torch.autocast(device_type=device.type, enabled=False):
                    losses = loss_fn(model, batch)
                    if (not isinstance(losses, Tensor) or losses.shape != (len(micro_indices),)
                            or losses.dtype != torch.float32 or losses.device != device):
                        raise ValueError("loss_fn must return float32 [microbatch size] on the model device")
                    assert_finite(losses, "per-reference losses")
                    if not losses.requires_grad:
                        raise RuntimeSafetyError("per-reference losses have no gradient graph")
                    (losses.sum() / actual_size).backward()
                step_loss += float(losses.detach().double().sum().item())
            nonzero = _gradient_check(model)
            assert_finite(model.state_dict(), "pre-step model")
            assert_finite(optimizer.state_dict(), "pre-step optimizer")
            optimizer.step()
            assert_finite(model.state_dict(), "post-step model")
            assert_finite(optimizer.state_dict(), "post-step optimizer")
        finally:
            # Also clears frozen parameters if a faulty callback created gradients.
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
        cursor = end
        global_step += 1
        steps += 1
        consumed += actual_size
        total_loss += step_loss
        state = StepState(epoch, seed, size, global_step, cursor,
                          math.ceil(cursor / effective_batch), indices, actual_size,
                          step_loss / actual_size, nonzero)
        if on_step is not None and on_step(state) is False:
            break
    return EpochResult(epoch, cursor, global_step, steps, consumed,
                       total_loss / consumed if consumed else None, cursor == size)
