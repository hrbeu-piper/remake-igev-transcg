"""CUDA-only single-reference training with resumable stage and sampler state."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from .runtime import (ResourceGuard, RunRegistry, RuntimeSafetyError, _write_json,
    assert_finite, canonical_hash, load_checkpoint, require_cuda_device, sha256_file)
from .train import _deployment
from .train_engine import run_epoch, _to_device
from .single_protocol import (read_json, require, validate_config, parent_checkpoint,
    validate_qualification, make_config, STAGES)
from .single_data import TransCGSingleDataset
from .single_network import SingleViewReMakeNetwork
from .single_objectives import SingleReferenceObjective, SingleLossConfig
from .monocular import RelativeDepthCache

def apply_backend(config):
    backend=config["backend"]
    torch.backends.cuda.matmul.allow_tf32=backend["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32=backend["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark=backend["cudnn_benchmark"]
    torch.backends.cudnn.deterministic=backend["cudnn_deterministic"]

def datasets(guard,config,manifest):
    cache=RelativeDepthCache(guard,config["mono"]["cache_dir"],
                             fingerprint=read_json(guard.path(config["mono"]["fingerprint_path"])))
    root=guard.path(manifest["dataset_root"])
    paths={r["sample_id"]:root/r["inputs"]["rgb"] for r in manifest["samples"] if r["split"] in ("train","val")}
    provider=cache.provider(paths)
    return tuple(TransCGSingleDataset(manifest,split,relative_depth_provider=provider,
        mask_protocol="gt_privileged",output_size=(480,640)) for split in ("train","val"))

def construct(config,guard,device):
    model=SingleViewReMakeNetwork(guard.path(config["vendor_root"]),stage=config["stage"],
        options=config["model_options"],freeze_bn_stats=True).to(device)
    opts=config["optimizer"]
    if config["stage"]=="fusion":
        groups=[{"params":[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("refiner.")],"lr":opts["lr"]},
                {"params":[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("remake.")],"lr":opts["decoder_lr"]}]
    else:
        groups=[{"params":[p for p in model.parameters() if p.requires_grad],"lr":opts["lr"]}]
    optimizer=torch.optim.AdamW(groups,weight_decay=opts["weight_decay"])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=config["scheduler"]["step_size"],gamma=config["scheduler"]["gamma"])
    return dict(model=model,optimizer=optimizer,scheduler=scheduler,scaler=None)

def initialize_parent(parts,parent):
    if parent is not None:
        parts["model"].load_state_dict(parent["state"],strict=True)

@torch.no_grad()
def validate(model,dataset,*,device,iterations,on_progress=None,should_stop=None):
    require(dataset.split=="val","Only isolated validation selects checkpoints")
    totals={"samples":0,"valid_pixels":0,"transparent_pixels":0,
            "valid_abs_error_sum_m":0.,"transparent_abs_error_sum_m":0.}
    before=model.training
    model.eval()
    try:
        for index in range(len(dataset)):
            if should_stop is not None and should_stop():
                raise InterruptedError("Stop requested during validation; repeat from saved boundary")
            batch=_to_device(default_collate([dataset[index]]),device)
            depth=model(batch["inputs"],iterations=iterations)["final_depth"]
            assert_finite(depth,"validation output")
            gt=batch["targets"]["depth_gt"]
            valid=batch["targets"]["gt_valid"].bool()&(gt>0)
            transparent=valid&batch["targets"]["transparent_mask"].bool()
            err=(depth-gt).abs().double()
            totals["valid_abs_error_sum_m"]+=float(err[valid].sum())
            totals["transparent_abs_error_sum_m"]+=float(err[transparent].sum())
            totals["valid_pixels"]+=int(valid.sum())
            totals["transparent_pixels"]+=int(transparent.sum())
            totals["samples"]+=1
            if on_progress and ((index+1)%25==0 or index+1==len(dataset)):
                on_progress(index+1)
        require(totals["valid_pixels"]>0 and totals["transparent_pixels"]>0,"No validation GT/ROI")
        totals.update(valid_mae_m=totals["valid_abs_error_sum_m"]/totals["valid_pixels"],
                      transparent_mae_mm=1000*totals["transparent_abs_error_sum_m"]/totals["transparent_pixels"])
        assert_finite(totals)
        return totals
    finally:
        model.train(before)

def run(config_path,*,resource_root,project_root,qualification_receipt):
    guard=ResourceGuard(resource_root,minimum_free_bytes=80*2**30)
    config=read_json(guard.path(config_path))
    checked=validate_config(config,guard,require_cache=True,require_dataset=True)
    apply_backend(config)
    revision=_deployment(guard,guard.path(project_root))
    qualification=validate_qualification(guard,qualification_receipt,config,revision)
    parent=parent_checkpoint(guard,config,config["manifest_sha256"])
    device=require_cuda_device(config["execution"]["device"])
    free,total=torch.cuda.mem_get_info(device)
    needed=int(qualification["peak_allocated_bytes"]*1.15)+512*2**20
    require(free>=needed,f"CUDA memory insufficient: free={free}, measured requirement={needed}; do not stop other processes")
    if os.getpgrp()!=os.getpid():
        os.setsid()
    stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    previous={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with RunRegistry(guard,config["run_id"],manifest_sha256=config["manifest_sha256"],
            config_sha256=checked["config_sha256"],execution_config=config["execution"],source_revision=revision) as registry:
            saved=registry.run_dir/"config.json"
            if saved.exists():
                require(canonical_hash(read_json(saved))==checked["config_sha256"],"Saved config differs")
            else:
                _write_json(guard,saved,config)
            _write_json(guard,registry.attempt_dir/"config.json",config)
            registry.record_event("qualified",{"receipt":str(qualification_receipt),"sha256":sha256_file(guard.path(qualification_receipt)),
                "verification_receipt_sha256":sha256_file(guard.path(config["verification_receipt"]))})
            seed=config["seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            parts=construct(config,guard,device)
            train,val=datasets(guard,config,checked["manifest"])
            objective=SingleReferenceObjective(SingleLossConfig(**config["loss"]),iterations=config["iterations"])
            offset=config["sampler_epoch_offset"]
            progress={"epoch":offset,"global_step":0,"next_batch":0,"accumulation_position":0,
                "best_metrics":{},"validation_count":0,"phase":"ready"}
            data_state={"sampler_epoch":offset,"sampler_seed":seed,"next_sample_index":0,"dataset_size":len(train)}
            checkpoint=registry.latest_checkpoint()
            if checkpoint is not None:
                loaded=load_checkpoint(checkpoint,guard=guard,expected_run_id=config["run_id"],
                    manifest_sha256=config["manifest_sha256"],config_sha256=checked["config_sha256"],**parts)
                progress,data_state=loaded["progress"],loaded["data_state"]
                require(data_state["dataset_size"]==len(train) and data_state["sampler_seed"]==seed,"Resume dataset/sampler differs")
                require(offset<=progress["epoch"]<=offset+config["epochs"],"Resume stage offset differs")
                registry.record_event("resumed",{"checkpoint":str(checkpoint),"progress":progress})
            else:
                initialize_parent(parts,parent)
                registry.record_event("initialized",{"source":None if parent is None else {k:v for k,v in parent.items() if k!="state"}})
            # Parent CPU tensors need not remain resident through the run.
            parent=None
            def save():
                return registry.save_checkpoint(**parts,progress=progress,data_state=data_state)
            if checkpoint is None:
                save()
            def pin_best():
                best=progress["best_metrics"]
                if progress["phase"]=="ready" and best.get("epoch")==progress["epoch"]-1 and best.get("global_step")==progress["global_step"]:
                    registry.publish_best_checkpoint()
            pin_best()
            start=time.monotonic()
            while progress["epoch"]<offset+config["epochs"] and not stopped:
                epoch=progress["epoch"]
                def on_step(state):
                    nonlocal progress,data_state
                    count=progress["validation_count"]
                    progress=state.progress(progress["best_metrics"]); progress.update(validation_count=count,phase="training")
                    data_state=state.data_state()
                    registry.update_progress(epoch=epoch,global_step=state.global_step,next_batch=state.next_batch,
                        validation_count=count,loss=state.loss_mean,
                        metrics={"stage_epoch":epoch-offset,"phase":"training","objective":objective.last_diagnostics})
                    if state.global_step%100==0 or stopped:
                        save(); registry.prune_checkpoints()
                    if state.global_step==1 or state.global_step%10==0:
                        print(json.dumps({"event":"train_step","stage":config["stage"],"epoch":epoch-offset,
                            "step":state.global_step,"loss":state.loss_mean,"elapsed_seconds":time.monotonic()-start}),flush=True)
                    return not stopped
                result=run_epoch(parts["model"],parts["optimizer"],train,objective,epoch=epoch,seed=seed,
                    start_sample=data_state["next_sample_index"],global_step=progress["global_step"],
                    effective_batch=8,physical_batch=1,device=device,on_step=on_step)
                if stopped:
                    save(); registry.prune_checkpoints(); registry.finish("stopped"); return
                require(result.completed,"Epoch ended unexpectedly")
                progress["phase"]="validation"; save(); registry.prune_checkpoints()
                previous_count=progress["validation_count"]
                def val_progress(count):
                    registry.update_progress(epoch=epoch,global_step=progress["global_step"],next_batch=progress["next_batch"],
                        validation_count=previous_count+count,metrics={"phase":"validation","samples":count,"stage_epoch":epoch-offset})
                try:
                    metrics=validate(parts["model"],val,device=device,iterations=config["iterations"],
                                     on_progress=val_progress,should_stop=lambda:stopped)
                except InterruptedError:
                    registry.finish("stopped"); return
                _write_json(guard,registry.attempt_dir/f"validation-epoch-{epoch-offset:03d}.json",metrics)
                best=progress["best_metrics"]
                if not best or metrics["transparent_mae_mm"]<best["transparent_mae_mm"]:
                    best={"transparent_mae_mm":metrics["transparent_mae_mm"],"epoch":epoch,"global_step":progress["global_step"]}
                parts["scheduler"].step()
                progress={**progress,"epoch":epoch+1,"next_batch":0,"best_metrics":best,"phase":"ready",
                          "validation_count":previous_count+len(val)}
                data_state={**data_state,"sampler_epoch":epoch+1,"next_sample_index":0}
                save(); pin_best(); registry.prune_checkpoints()
                registry.update_progress(epoch=epoch+1,global_step=progress["global_step"],next_batch=0,
                                         validation_count=progress["validation_count"],metrics=metrics)
                registry.record_event("epoch_complete",{"stage_epoch":epoch-offset+1,"metrics":metrics,"best":best})
                print(json.dumps({"event":"epoch_complete","stage":config["stage"],"epoch":epoch-offset+1,"metrics":metrics}),flush=True)
            if progress["epoch"]==offset+config["epochs"]:
                receipt={"kind":"single_view_training_completion","complete":True,"stage":config["stage"],
                    "epochs":config["epochs"],"sampler_epoch_offset":offset,"config_sha256":checked["config_sha256"],
                    "manifest_sha256":config["manifest_sha256"],"validation_samples_per_epoch":len(val),
                    "training_samples_per_epoch":len(train),"source_revision":revision,"progress":progress,
                    "best_checkpoint":registry.registry["best_checkpoint"]}
                _write_json(guard,registry.run_dir/"completion.json",receipt)
                registry.finish("completed")
            else:
                save(); registry.prune_checkpoints(); registry.finish("stopped")
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--resource-root",type=Path,required=True)
    p.add_argument("--project-root",type=Path,required=True)
    p.add_argument("--qualification-receipt",type=Path,required=True)
    a=p.parse_args()
    run(a.config,resource_root=a.resource_root,project_root=a.project_root,qualification_receipt=a.qualification_receipt)

if __name__=="__main__":
    main()
