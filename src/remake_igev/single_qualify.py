"""Real single-view CUDA learning/restore qualification on eight fixed train references."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import Subset,default_collate
from .runtime import (ResourceGuard,RunRegistry,_write_json,assert_finite,canonical_hash,
    capture_rng,load_checkpoint,require_cuda_device,sha256_file)
from .qualify import _equal,loss_windows
from .train import _deployment
from .train_engine import run_epoch,_to_device
from .single_protocol import read_json,require,validate_config,parent_checkpoint,validate_qualification
from .single_train import construct,datasets,initialize_parent,apply_backend
from .single_objectives import SingleLossConfig,SingleReferenceObjective

def fixed_subset(train):
    require(train.split=="train","Qualification uses train only")
    # Independent of file availability, GT values or observed training losses.
    selected=sorted(range(len(train)),key=lambda i:train.rows[i]["sample_id"])[:8]
    require(len(selected)==8,"Qualification requires eight fixed references")
    return Subset(train,selected),selected

def source_identity(guard,manifest,rows):
    root=guard.path(manifest["dataset_root"])
    result={}
    for row in rows:
        files=[row["inputs"]["rgb"],row["inputs"]["raw_depth"],row["inputs"]["transparent_mask"],
               row["supervision"]["depth_gt"]]
        for name in files:
            path=guard.path(root/name)
            result[name]={"bytes":path.stat().st_size,"sha256":sha256_file(path)}
    return result

def restore_check(parts,config,guard,device,batch,checkpoint,registry,diag_sha,progress,data_state):
    before=capture_rng()
    parts["model"].eval()
    with torch.no_grad():
        expected=parts["model"](batch["inputs"],iterations=config["iterations"])["final_depth"].cpu()
    _equal(before,capture_rng(),"eval RNG")
    restored=construct(config,guard,device)
    loaded=load_checkpoint(checkpoint,guard=guard,expected_run_id=registry.run_id,
        manifest_sha256=config["manifest_sha256"],config_sha256=diag_sha,**restored)
    _equal(progress,loaded["progress"],"progress")
    _equal(data_state,loaded["data_state"],"sampler")
    _equal(before,capture_rng(),"restored RNG")
    for key in ("model","optimizer","scheduler"):
        _equal(parts[key].state_dict(),restored[key].state_dict(),key)
    restored["model"].eval()
    with torch.no_grad():
        actual=restored["model"](batch["inputs"],iterations=config["iterations"])["final_depth"].cpu()
    assert_finite(actual)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    _equal(before,capture_rng(),"post restore RNG")
    return restored

def run(config_path,*,resource_root,project_root,output):
    guard=ResourceGuard(resource_root,minimum_free_bytes=80*2**30)
    output=guard.path(output)
    require(not output.exists() and not Path(str(output)+".log").exists(),"Qualification output exists; retain it and use a new attempt")
    guard.mkdir(output.parent)
    receipt={"schema_version":1,"kind":"single_view_cuda_qualification","mode":"diagnostic",
        "passed":False,"formal_parent_eligible":False,"device_type":"cuda","sample_split":"train",
        "qualification_scope":"fixed_eight_training_references","formal_training_authorization":False,
        "reference_samples":8,"test_samples_used":0,"optimization_steps":0,
        "real_batch_forward_backward":False,"checkpoint_restore_verified":False,"finite_updates":False}
    losses=[]; registry=None; first_state=None
    log=Path(str(output)+".log").open("x",encoding="utf-8")
    def emit(event,**details):
        message=json.dumps({"event":event,**details},ensure_ascii=False,allow_nan=False)
        print(message,flush=True); log.write(message+"\n"); log.flush()
    try:
        config=read_json(guard.path(config_path))
        checked=validate_config(config,guard)
        apply_backend(config)
        revision=_deployment(guard,guard.path(project_root))
        device=require_cuda_device(config["execution"]["device"])
        free,total=torch.cuda.mem_get_info(device)
        require(free>=6*2**30,f"Less than 6GiB CUDA memory free ({free}); do not stop other jobs")
        receipt.update(stage=config["stage"],config_sha256=checked["config_sha256"],source_revision=revision,
            execution_config_sha256=canonical_hash(config["execution"]),manifest_sha256=config["manifest_sha256"],
            device_name=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,initial_cuda_free_bytes=free)
        receipt["deployment_sha256"]=sha256_file(guard.path(project_root)/"deployment.json")
        receipt["backend"]=dict(config["backend"])
        seed=config["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        train,_=datasets(guard,config,checked["manifest"])
        subset,indices=fixed_subset(train)
        rows=[train.rows[i] for i in indices]
        receipt["sample_ids"]=[row["sample_id"] for row in rows]
        receipt["source_files"]=source_identity(guard,checked["manifest"],rows)
        parent=parent_checkpoint(guard,config,config["manifest_sha256"])
        diag={"kind":"single_view_qualification_config","mode":"diagnostic",
            "origin_config_sha256":checked["config_sha256"],"sample_ids":receipt["sample_ids"],"steps":200,
            "source_files":receipt["source_files"],"source_revision":revision}
        diag_sha=canonical_hash(diag)
        run_id="sv-qualification-"+canonical_hash(str(output))[:20]
        if os.getpgrp()!=os.getpid(): os.setsid()
        with RunRegistry(guard,run_id,manifest_sha256=config["manifest_sha256"],config_sha256=diag_sha,
                        execution_config=config["execution"],source_revision=revision) as registry:
            require(len(registry.registry["attempts"])==1,"Qualification retry requires a fresh path")
            _write_json(guard,registry.run_dir/"config.json",diag)
            receipt.update(run_id=run_id,registry_path=str(registry.registry_path),diagnostic_config_sha256=diag_sha)
            torch.cuda.reset_peak_memory_stats(device)
            parts=construct(config,guard,device); initialize_parent(parts,parent); parent=None
            objective=SingleReferenceObjective(SingleLossConfig(**config["loss"]),iterations=config["iterations"])
            gradient_groups=set(); hooks=[]
            def install_hooks(model):
                def hook_for(name):
                    def hook(grad):
                        if bool(torch.count_nonzero(grad)):
                            gradient_groups.add("refiner" if name.startswith("refiner.") else "remake")
                            for branch,label in enumerate(("global","medium","fine")):
                                if name.startswith(f"refiner.aggregators.{branch}."):
                                    gradient_groups.add(label)
                            for label in ("gru4","gru8","gru16","gate"):
                                if name.startswith(f"refiner.{label}."):
                                    gradient_groups.add(label)
                        return grad
                    return hook
                return [p.register_hook(hook_for(name)) for name,p in model.named_parameters() if p.requires_grad]
            hooks=install_hooks(parts["model"])
            start=time.monotonic()
            last_state=None
            def save(state):
                registry.save_checkpoint(**parts,progress=state.progress(),data_state=state.data_state())
                registry.prune_checkpoints()
            def on_step(state):
                nonlocal last_state
                last_state=state; losses.append(state.loss_mean)
                # Use the full dataset equivalent; the eight-image diagnostic
                # is not allowed to decay LR 200 times in 200 updates.
                if state.global_step%math.ceil(len(train)/8)==0: parts["scheduler"].step()
                registry.update_progress(epoch=state.epoch,global_step=state.global_step,next_batch=state.next_batch,
                    loss=state.loss_mean,validation_count=0,metrics=objective.last_diagnostics)
                if state.global_step==1 or state.global_step%25==0:
                    save(state)
                emit("step",step=state.global_step,loss=state.loss_mean,elapsed_seconds=time.monotonic()-start,
                     nonzero_gradient_parameters=state.nonzero_gradient_parameters)
            epoch=config["sampler_epoch_offset"]; step=0; cursor=0
            while step<200:
                result=run_epoch(parts["model"],parts["optimizer"],subset,objective,epoch=epoch,seed=seed,
                    start_sample=cursor,global_step=step,effective_batch=8,physical_batch=1,
                    device=device,on_step=on_step,max_steps=1 if step==0 else 200-step)
                step,cursor=result.global_step,result.next_sample_index
                if step==1 and not receipt["checkpoint_restore_verified"]:
                    batch=_to_device(default_collate([subset[0]]),device)
                    for handle in hooks: handle.remove()
                    restored=restore_check(parts,config,guard,device,batch,registry.latest_checkpoint(),registry,
                        diag_sha,last_state.progress(),last_state.data_state())
                    parts=restored; hooks=install_hooks(parts["model"]); del batch
                    gc.collect(); torch.cuda.empty_cache()
                    receipt.update(real_batch_forward_backward=True,checkpoint_restore_verified=True,
                        state_and_rng_restore="exact",eval_atol=1e-6,eval_rtol=1e-5)
                    emit("checkpoint_restore_verified")
                if result.completed: epoch+=1; cursor=0
            for handle in hooks: handle.remove()
            torch.cuda.synchronize()
            initial,final=loss_windows(losses)
            needed={"refiner","remake"} if config["stage"]=="fusion" else {"refiner"} if config["stage"]=="refine" else {"remake"}
            if config["stage"] in ("refine","fusion"):
                needed|={"global","medium","fine","gru4","gru8","gru16","gate"}
            require(needed.issubset(gradient_groups),"A target trainable branch had no nonzero gradient")
            require(source_identity(guard,checked["manifest"],rows)==receipt["source_files"],"Qualification source changed")
            elapsed=time.monotonic()-start
            receipt.update(passed=True,finite_updates=True,optimization_steps=200,initial_mean_loss=initial,
                final_mean_loss=final,loss_window_steps=20,nonzero_gradient_groups=sorted(gradient_groups),
                elapsed_seconds=elapsed,seconds_per_optimizer_step=elapsed/200,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                latest_checkpoint=registry.registry["latest_checkpoint"])
            candidate=registry.attempt_dir/"qualification-candidate.json"
            _write_json(guard,candidate,receipt)
            require(_deployment(guard,guard.path(project_root))==revision,"Qualification source changed")
            validate_qualification(guard,candidate,config,revision)
            registry.finish("completed"); _write_json(guard,output,receipt)
            emit("passed",initial_mean_loss=initial,final_mean_loss=final,seconds_per_optimizer_step=elapsed/200)
            return receipt
    except BaseException as error:
        receipt.update(passed=False,optimization_steps=len(losses),error=f"{type(error).__name__}: {error}")
        if losses:
            receipt.update(initial_mean_loss=float(np.mean(losses[:20])),final_mean_loss=float(np.mean(losses[-20:])))
        if registry is not None and registry.registry is not None:
            receipt["latest_checkpoint"]=registry.registry.get("latest_checkpoint")
        _write_json(guard,output,receipt)
        emit("failed",error=receipt["error"],traceback=traceback.format_exc())
        raise
    finally:
        log.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","resource-root","project-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.config,resource_root=a.resource_root,project_root=a.project_root,output=a.output)

if __name__=="__main__":
    main()
