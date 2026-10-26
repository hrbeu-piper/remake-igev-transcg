"""One bounded, detached preparation/training chain; no desktop reminder task."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from .runtime import ResourceGuard,_write_json,canonical_hash,process_identity,process_matches,sha256_file
from .single_protocol import read_json,require,make_config,verify_dataset_receipt,validate_qualification
from .train import _deployment

POLL_SECONDS=15
STOP_GRACE_SECONDS=120

def completed_stage(guard,run_dir,config):
    directory=guard.path(run_dir)
    record=read_json(directory/"completion.json")
    registry=read_json(directory/"registry.json")
    expected_step=((29861+7)//8)*config["epochs"]
    expected_epoch=config["sampler_epoch_offset"]+config["epochs"]
    require(record.get("kind")=="single_view_training_completion" and record.get("complete") is True
        and record.get("stage")==config["stage"] and record.get("config_sha256")==canonical_hash(config)
        and record.get("manifest_sha256")==config["manifest_sha256"]
        and record.get("epochs")==config["epochs"] and record.get("sampler_epoch_offset")==config["sampler_epoch_offset"]
        and record.get("training_samples_per_epoch")==29861 and record.get("validation_samples_per_epoch")==4330,
        "Stage completion protocol/budget differs")
    progress=record.get("progress",{})
    require(progress.get("epoch")==expected_epoch and progress.get("global_step")==expected_step
        and progress.get("validation_count")==4330*config["epochs"] and progress.get("phase")=="ready"
        and progress.get("next_batch")==0,"Stage did not finish its complete training/validation budget")
    require(registry.get("config_sha256")==canonical_hash(config) and registry.get("attempts")
        and registry["attempts"][-1].get("status")=="completed","Trainer stopped or failed rather than completed")
    require(registry["attempts"][-1].get("progress",{}).get("global_step")==expected_step,
            "Registry does not confirm the final update")
    best=record.get("best_checkpoint")
    require(isinstance(best,dict) and best==registry.get("best_checkpoint"),"Validation-selected best differs")
    path=guard.path(best["path"])
    require(path.is_relative_to(directory) and sha256_file(path)==best["sha256"],"Best checkpoint is missing or changed")
    return record

def run(resource_root,project_root,control_dir,*,wait_hours=12):
    guard=ResourceGuard(resource_root,minimum_free_bytes=80*2**30)
    project=guard.path(project_root); control=guard.mkdir(control_dir)
    lock=(control/"pipeline.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=control/"pipeline-status.json"
    previous=read_json(state_path) if state_path.exists() else {}
    require(previous.get("state")!="completed","Pipeline already completed")
    prior_child=previous.get("child")
    require(not prior_child or not process_matches(prior_child),"A previous pipeline child is still running")
    if os.getpgrp()!=os.getpid(): os.setsid()
    revision=_deployment(guard,project)
    deployment_sha=sha256_file(project/"deployment.json")
    require(not previous or previous.get("deployment_sha256")==deployment_sha,
            "Pipeline deployment changed across resume; preserve this control and requalify explicitly")
    state={"kind":"single_view_training_pipeline","source_revision":revision,"state":"running",
        "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"owner":process_identity(),
        "phase":"initialization","formal_training_started":bool(previous.get("formal_training_started",False)),
        "deployment_sha256":deployment_sha,"history":previous.get("history",[])}
    stop=False; child=None; child_owner=None
    def request_stop(*_):
        nonlocal stop
        stop=True
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def frozen_source():
        require(sha256_file(project/"deployment.json")==deployment_sha
                and _deployment(guard,project)==revision,"Frozen source/deployment changed during pipeline")
    def stop_child():
        if child is None or child.poll() is not None:
            return
        if child_owner is not None:
            require(process_matches(child_owner),"Child identity changed; refuse to signal another process")
        else:
            # Popen still owns an unreaped direct child if identity capture failed.
            status=Path(f"/proc/{child.pid}/status").read_text()
            require(f"PPid:\t{os.getpid()}\n" in status and os.getpgid(child.pid)==child.pid,
                    "Unidentified child is not our independent process group")
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            require(child.poll() is None and (child_owner is None or process_matches(child_owner)),
                    "Child ownership changed during stop grace")
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=30)
    def publish(event=None,**extra):
        state.update(extra,updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
        _write_json(guard,state_path,state)
        if event:
            message={"at":state["updated_at"],"event":event,**extra}
            with (control/"pipeline-events.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(message,ensure_ascii=False)+"\n")
            print(json.dumps(message,ensure_ascii=False),flush=True)
    def execute(phase,arguments):
        nonlocal child,child_owner
        require(not stop,"Pipeline stop requested")
        frozen_source()
        attempt=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        logs=guard.mkdir(control/"logs")
        log_path=logs/(phase+"-"+attempt+".log")
        command=[sys.executable,"-m",*arguments]
        with log_path.open("xb") as log:
            environment=dict(os.environ,PYTHONPATH=str(project/"src"),PYTHONDONTWRITEBYTECODE="1",
                             OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
                             TMPDIR=str(guard.mkdir(guard.root/"tmp/remake_igev-checks")))
            child=subprocess.Popen(command,cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            # Capture identity from the child, without assuming a PID is enough.
            stat=Path(f"/proc/{child.pid}/stat").read_text()
            owner={"pid":child.pid,"start_ticks":stat.rsplit(")",1)[1].split()[19],
                "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "hostname":os.uname().nodename,"pgid":os.getpgid(child.pid),
                "cmdline":Path(f"/proc/{child.pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")}
            child_owner=owner
            publish("phase_started",phase=phase,child=owner,command=command,log_path=str(log_path))
            notified_stop=False
            while child.poll() is None:
                if stop and not notified_stop:
                    require(process_matches(owner),"Child identity changed before requested stop")
                    stop_child(); notified_stop=True
                frozen_source()
                if arguments[0]=="remake_igev.single_train":
                    training_config=read_json(Path(arguments[arguments.index("--config")+1]))
                    registry_path=root/"experiment/remake_igev"/training_config["run_id"]/"registry.json"
                    if registry_path.exists():
                        registry=read_json(registry_path)
                        current=registry["attempts"][-1]
                        if current["owner"]["pid"]==child.pid:
                            state["training_progress"]=current.get("progress",{})
                            if current.get("progress",{}).get("global_step",0)>0:
                                state["formal_training_started"]=True
                publish()
                time.sleep(POLL_SECONDS)
            code=child.returncode
        state["history"].append({"phase":phase,"exit_code":code,"log_path":str(log_path),"source_revision":revision})
        publish("phase_finished",child=None,exit_code=code)
        require(code==0 and not stop,f"{phase} exited {code}; preserve log/checkpoint, diagnose before another attempt")
        child=None; child_owner=None
    root=guard.root
    baseline=read_json(control/"baseline-config.json")
    verification=guard.path(baseline["verification_receipt"])
    common_cache=["remake_igev.single_cache","--manifest",baseline["manifest_path"],
        "--project-root",str(project),"--resource-root",str(root),"--cache-dir",baseline["mono"]["cache_dir"],
        "--fingerprint-out",baseline["mono"]["fingerprint_path"],"--verification-receipt",str(verification)]
    try:
        publish("pipeline_started")
        # Qualification is narrowly scoped to eight real train references. Its
        # success cannot satisfy the independent full-data/cache gates below.
        def qualify_stage(stage,config_path,config):
            directory=guard.mkdir(control/"qualifications")
            receipt=directory/(stage+".json")
            if receipt.exists():
                validate_qualification(guard,receipt,config,revision)
                publish("qualification_reused",phase="qualification_"+stage)
            else:
                execute("qualification_"+stage,["remake_igev.single_qualify","--config",str(config_path),
                    "--resource-root",str(root),"--project-root",str(project),"--output",str(receipt)])
                validate_qualification(guard,receipt,config,revision)
            return receipt
        baseline_qual=qualify_stage("baseline",control/"baseline-config.json",baseline)
        publish("waiting_for_complete_dataset",phase="wait_full_dataset")
        deadline=time.monotonic()+wait_hours*3600
        while not verification.exists():
            require(not stop,"Stop requested while waiting for data")
            require(time.monotonic()<deadline,"Data verification wait exceeded its explicit time budget")
            restore_path=root/"tmp/transcg-restore-20261026/status.json"
            if restore_path.exists():
                restore=read_json(restore_path)
                require(restore.get("state")!="failed","Dataset restoration failed; diagnose it before training")
                state["dataset_restore"]={k:restore.get(k) for k in ("state","stage","current_archive","downloaded_chunks","total_chunks")}
            publish()
            time.sleep(30)
        verify_dataset_receipt(guard,verification,root/"datasets/transcg")
        audit=control/"train-depth-range.json"
        if not audit.exists():
            execute("train_depth_range",common_cache+["--status-out",str(audit),"--audit-depth-range"])
        result=read_json(audit)
        require(result.get("complete_train") is True and result.get("reference_count")==29861
                and result.get("manifest_sha256")==baseline["manifest_sha256"],"Train range audit incomplete/stale")
        publish("train_range_frozen",phase="train_depth_range",train_range_coverage=result["coverage_fraction"],
                candidate_depth_range_m=[.1,2.],GT_clipped=False)
        cache_completion=guard.path(baseline["mono"]["completion_receipt"])
        # Replays cache checks on resume; the cache's own source-bound hits avoid
        # expensive inference and do not trust a stale completed cursor.
        execute("mono_cache",common_cache+["--status-out",str(cache_completion)])
        paths={"baseline":control/"baseline-config.json"}
        configs={"baseline":baseline}
        run_dirs={"baseline":root/"experiment/remake_igev"/baseline["run_id"]}
        for stage in ("baseline","baseline_continue","refine","fusion"):
            if stage!="baseline":
                parent_stage="refine" if stage=="fusion" else "baseline"
                run_id=f"sv-{stage.replace('_','-')}-20261026-v1"
                config=make_config(guard=guard,stage=stage,run_id=run_id,manifest_path=baseline["manifest_path"],
                    fingerprint_path=baseline["mono"]["fingerprint_path"],cache_dir=baseline["mono"]["cache_dir"],
                    cache_receipt=cache_completion,verification_receipt=verification,vendor_root=baseline["vendor_root"],
                    parent_run=run_dirs[parent_stage])
                path=control/(stage+"-config.json")
                if path.exists(): require(read_json(path)==config,"Stored stage configuration changed")
                else: _write_json(guard,path,config)
                configs[stage]=config; paths[stage]=path
                run_dirs[stage]=root/"experiment/remake_igev"/run_id
            config=configs[stage]; path=paths[stage]
            completion=run_dirs[stage]/"completion.json"
            if completion.exists():
                completed_stage(guard,run_dirs[stage],config)
                state["formal_training_started"]=True
                publish("completed_stage_reused",phase="train_"+stage)
                continue
            receipt=baseline_qual if stage=="baseline" else qualify_stage(stage,path,config)
            execute("train_"+stage,["remake_igev.single_train","--config",str(path),"--resource-root",str(root),
                "--project-root",str(project),"--qualification-receipt",str(receipt)])
            completed_stage(guard,run_dirs[stage],config)
            state["formal_training_started"]=True
        publish("training_budget_completed",state="completed",phase="training_complete",
                evaluation_state="pending_frozen_model_capacity_and_archive_sequence")
    except BaseException as error:
        stop_child()
        publish("failed",state="stopped" if stop else "failed",error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc())
        raise
    finally:
        stop_child()
        for s,handler in handlers.items(): signal.signal(s,handler)
        lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("resource-root","project-root","control-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--wait-hours",type=float,default=12)
    a=p.parse_args()
    require(0<a.wait_hours<=24,"Wait budget must be positive and at most 24 hours")
    run(a.resource_root,a.project_root,a.control_dir,wait_hours=a.wait_hours)

if __name__=="__main__":
    main()
