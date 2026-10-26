"""Frozen single-reference training protocol, without any calibration gate."""
from __future__ import annotations
import json
from pathlib import Path
import re
import socket
from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash, sha256_file, assert_finite

PROTOCOL = "single_view_multirange_v1"
STAGES = {
    "baseline": dict(epochs=40, offset=0, lr=1e-3, decoder_lr=None, gamma=.5, parent=None),
    "refine": dict(epochs=5, offset=0, lr=1e-4, decoder_lr=None, gamma=1., parent="baseline"),
    "fusion": dict(epochs=15, offset=5, lr=1e-4, decoder_lr=1e-5, gamma=.5, parent="refine"),
    "baseline_continue": dict(epochs=20, offset=0, lr=1e-4, decoder_lr=None, gamma=.5, parent="baseline"),
}
SEED=20261024
BACKEND={"cuda_matmul_allow_tf32":False,"cudnn_allow_tf32":False,
         "cudnn_benchmark":False,"cudnn_deterministic":False}

def require(condition, message):
    if not condition:
        raise RuntimeSafetyError(message)

def read_json(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value,dict), "Expected a JSON object")
    return value

def verify_dataset_receipt(guard, receipt_path, dataset_root):
    from verify_autodl_transcg import validate_verified_receipt
    path=guard.path(receipt_path)
    root=guard.path(dataset_root)
    require(root==guard.root/"datasets/transcg", "Unexpected current-instance dataset")
    result=validate_verified_receipt(path,root)
    require(isinstance(result,dict), "Dataset validator returned no verified evidence")
    return result

def validate_config(config, guard, *, require_cache=False, require_dataset=False):
    from .single_manifest import validate_manifest
    from .single_network import DEFAULT_SINGLE_OPTIONS
    from .single_objectives import SingleLossConfig
    from dataclasses import asdict
    stage=config.get("stage")
    require(stage in STAGES, "Unknown single-view training stage")
    spec=STAGES[stage]
    expected={"schema_version":1,"kind":PROTOCOL,"mode":"formal",
        "seed":SEED,"epochs":spec["epochs"],"sampler_epoch_offset":spec["offset"],
        "output_hw":[480,640],"mask_protocol":"gt_privileged","effective_batch":8,
        "iterations":8,"checkpoint_every_steps":100,"loss":asdict(SingleLossConfig()),
        "model_options":dict(DEFAULT_SINGLE_OPTIONS),"backend":dict(BACKEND),
        "execution":{"device":"cuda:0","precision":"fp32","physical_batch":1,
                     "accumulation_steps":8,"num_workers":0,"augmentation":"identity",
                     "freeze_bn_stats":True},
        "optimizer":{"kind":"AdamW","lr":spec["lr"],"decoder_lr":spec["decoder_lr"],"weight_decay":1e-4},
        "scheduler":{"kind":"StepLR","step_size":10,"gamma":spec["gamma"]}}
    require(all(config.get(k)==v for k,v in expected.items()), "Frozen single-view protocol/budget differs")
    require(re.fullmatch(r"sv-[a-z0-9-]{1,90}",config.get("run_id","")) is not None,"Invalid run ID")
    manifest_path=guard.path(config["manifest_path"])
    require(sha256_file(manifest_path)==config["manifest_sha256"],"Manifest hash differs")
    manifest=read_json(manifest_path)
    validate_manifest(manifest)
    require(manifest["dataset_root"]==str(guard.root/"datasets/transcg"),"Manifest dataset differs")
    mono=config["mono"]
    fp_path=guard.path(mono["fingerprint_path"])
    require(sha256_file(fp_path)==mono["fingerprint_file_sha256"],"Monocular fingerprint file changed")
    fp=read_json(fp_path)
    from .monocular import _fingerprint
    _fingerprint(fp,allow_cpu_checks=False)
    require(fp.get("execution_mode")=="cuda" or str(fp.get("execution_mode","")).startswith("cuda"),
            "Production prior must be generated on CUDA")
    require(fp.get("cache_storage",{}).get("output_hw")==[480,640],"Prior must use the fixed grid")
    require(canonical_hash(fp)==mono["fingerprint_sha256"],"Prior fingerprint differs")
    guard.path(config["vendor_root"]); guard.path(mono["cache_dir"])
    if require_dataset:
        verify_dataset_receipt(guard,config["verification_receipt"],manifest["dataset_root"])
        for relative,expected_sha in manifest["verification"]["metadata_sha256"].items():
            require(sha256_file(guard.path(Path(manifest["dataset_root"])/relative))==expected_sha,
                    "Actual metadata changed since the frozen split")
    if require_cache:
        cache=read_json(guard.path(mono["completion_receipt"]))
        require(cache.get("complete") is True and cache.get("manifest_sha256")==config["manifest_sha256"],
                "Full train/val cache is incomplete or belongs to another manifest")
        require(cache.get("fingerprint_sha256")==mono["fingerprint_sha256"],"Completed cache fingerprint differs")
        require(cache.get("split_counts")=={"train":29861,"val":4330},"Cache sample coverage differs")
        require(Path(cache["cache_dir"]).resolve()==guard.path(mono["cache_dir"]),"Cache root differs")
        require(cache.get("kind")=="single_view_mono_cache" and cache.get("scope")=="full_train_val"
                and cache.get("status")=="completed" and cache.get("processed")==34191,
                "Full cache completion scope differs")
        index_path=guard.path(cache["cache_index_path"])
        require(sha256_file(index_path)==cache["cache_index_sha256"],"Cache index changed")
        index=read_json(index_path)
        expected_rows=[row for row in manifest["samples"] if row["split"] in ("train","val")]
        require([(item["sample_id"],item["split"],item["rgb"]) for item in index.get("entries",[])]
                ==[(row["sample_id"],row["split"],row["inputs"]["rgb"]) for row in expected_rows],
                "Cache index reference coverage differs")
        require(index.get("fingerprint_sha256")==mono["fingerprint_sha256"]
                and index.get("manifest_sha256")==config["manifest_sha256"],"Cache index provenance differs")
        for item in index["entries"]:
            entry_path=guard.path(item["cache_path"])
            require(entry_path.is_relative_to(guard.path(mono["cache_dir"])) and entry_path.is_file(),
                    "A registered cache entry is missing or outside its namespace")
    initialization=config["initialization"]
    if spec["parent"] is None:
        require(initialization=={"kind":"fresh"},"Baseline cannot use official/diagnostic trained weights")
    else:
        require(initialization.get("kind")=="completed_single_view_best","Stage requires a completed formal parent")
        guard.path(initialization["run_dir"])
    return {"config_sha256":canonical_hash(config),"manifest":manifest,"fingerprint":fp}

def parent_checkpoint(guard, config, manifest_sha):
    stage=config["stage"]
    if STAGES[stage]["parent"] is None:
        return None
    directory=guard.path(config["initialization"]["run_dir"])
    require(directory.parent==guard.root/"experiment/remake_igev","Parent outside this task registry")
    parent=read_json(directory/"config.json")
    summary=validate_config(parent,guard)
    registry=read_json(directory/"registry.json")
    require(parent["stage"]==STAGES[stage]["parent"],"Wrong parent stage")
    require(parent["manifest_sha256"]==manifest_sha and parent["mono"]["fingerprint_sha256"]==config["mono"]["fingerprint_sha256"],
            "Parent data/prior differs")
    require(registry["attempts"][-1]["status"]=="completed" and registry["config_sha256"]==summary["config_sha256"],
            "Parent not completed or config changed")
    completion=read_json(directory/"completion.json")
    require(completion.get("complete") is True and completion.get("kind")=="single_view_training_completion"
            and completion.get("config_sha256")==summary["config_sha256"]
            and completion.get("epochs")==parent["epochs"]
            and completion.get("training_samples_per_epoch")==29861
            and completion.get("validation_samples_per_epoch")==4330,
            "Parent full-budget completion evidence differs")
    best=registry.get("best_checkpoint")
    require(isinstance(best,dict),"Parent has no validation-selected best")
    path=guard.path(best["path"])
    require(path.is_relative_to(directory) and sha256_file(path)==best["sha256"],"Parent checkpoint identity differs")
    import torch
    payload=torch.load(path,map_location="cpu",weights_only=False)
    require(payload["run_id"]==parent["run_id"] and payload["manifest_sha256"]==manifest_sha
            and payload["config_sha256"]==summary["config_sha256"] and payload["attempt_id"]==best["attempt_id"],
            "Parent checkpoint metadata differs")
    assert_finite(payload,"parent checkpoint")
    require(payload["progress"]["global_step"]==best["global_step"],"Best checkpoint step differs")
    return {"path":str(path),"sha256":best["sha256"],"run_id":parent["run_id"],"stage":parent["stage"],
            "selected_epoch":payload["progress"]["best_metrics"].get("epoch"),"state":payload["states"]["model"]}

def make_config(*, guard, stage, run_id, manifest_path, fingerprint_path, cache_dir,
                cache_receipt, verification_receipt, vendor_root, parent_run=None):
    from .single_network import DEFAULT_SINGLE_OPTIONS
    from .single_objectives import SingleLossConfig
    from dataclasses import asdict
    spec=STAGES[stage]
    require((parent_run is None)==(spec["parent"] is None),"Parent argument does not match stage")
    fp=read_json(guard.path(fingerprint_path))
    return {"schema_version":1,"kind":PROTOCOL,"mode":"formal","stage":stage,"run_id":run_id,
        "seed":SEED,"epochs":spec["epochs"],"sampler_epoch_offset":spec["offset"],
        "output_hw":[480,640],"mask_protocol":"gt_privileged","effective_batch":8,"iterations":8,
        "checkpoint_every_steps":100,"vendor_root":str(guard.path(vendor_root)),
        "manifest_path":str(guard.path(manifest_path)),"manifest_sha256":sha256_file(guard.path(manifest_path)),
        "verification_receipt":str(guard.path(verification_receipt)),
        "mono":{"cache_dir":str(guard.path(cache_dir)),"fingerprint_path":str(guard.path(fingerprint_path)),
                "fingerprint_file_sha256":sha256_file(guard.path(fingerprint_path)),"fingerprint_sha256":canonical_hash(fp),
                "completion_receipt":str(guard.path(cache_receipt))},
        "initialization":{"kind":"fresh"} if parent_run is None else {"kind":"completed_single_view_best","run_dir":str(guard.path(parent_run))},
        "loss":asdict(SingleLossConfig()),"model_options":dict(DEFAULT_SINGLE_OPTIONS),"backend":dict(BACKEND),
        "execution":{"device":"cuda:0","precision":"fp32","physical_batch":1,"accumulation_steps":8,
                     "num_workers":0,"augmentation":"identity","freeze_bn_stats":True},
        "optimizer":{"kind":"AdamW","lr":spec["lr"],"decoder_lr":spec["decoder_lr"],"weight_decay":1e-4},
        "scheduler":{"kind":"StepLR","step_size":10,"gamma":spec["gamma"]}}

def validate_qualification(guard,path,config,revision):
    receipt=read_json(guard.path(path))
    assert_finite(receipt,"qualification receipt")
    required={"kind":"single_view_cuda_qualification","passed":True,"stage":config["stage"],
        "config_sha256":canonical_hash(config),"source_revision":revision,"device_type":"cuda",
        "real_batch_forward_backward":True,"checkpoint_restore_verified":True,"finite_updates":True,
        "sample_split":"train","test_samples_used":0,"optimization_steps":200,"reference_samples":8,
        "qualification_scope":"fixed_eight_training_references","formal_training_authorization":False,
        "mode":"diagnostic","formal_parent_eligible":False,"backend":config["backend"],
        "manifest_sha256":config["manifest_sha256"],
        "deployment_sha256":sha256_file(guard.path(Path(config["vendor_root"]).parent.parent/"deployment.json"))}
    require(all(receipt.get(k)==v for k,v in required.items()),"Missing/current single-view CUDA qualification")
    require(receipt["initial_mean_loss"]>0 and 0<=receipt["final_mean_loss"]<=.8*receipt["initial_mean_loss"],
            "Qualification loss-decrease condition failed")
    require(receipt["execution_config_sha256"]==canonical_hash(config["execution"]),"Qualified execution settings differ")
    from .single_manifest import load_manifest,required_files
    manifest=load_manifest(guard.path(config["manifest_path"]))
    rows=sorted((row for row in manifest["samples"] if row["split"]=="train"),key=lambda row:row["sample_id"])[:8]
    require(receipt.get("sample_ids")==[row["sample_id"] for row in rows],"Qualification reference selection differs")
    require(set(receipt.get("source_files",{}))=={name for row in rows for name in required_files(row)},
            "Qualification source coverage differs")
    needed={"refiner","remake"} if config["stage"]=="fusion" else {"refiner"} if config["stage"]=="refine" else {"remake"}
    if config["stage"] in ("refine","fusion"):
        needed|={"global","medium","fine","gru4","gru8","gru16","gate"}
    require(needed.issubset(set(receipt.get("nonzero_gradient_groups",[]))),"Qualification target gradients missing")
    require(type(receipt.get("peak_allocated_bytes")) is int and receipt["peak_allocated_bytes"]>0,
            "Missing actual GPU memory measurement")
    # Qualification may precede the full transfer. It only certifies this fixed
    # batch/implementation; training separately requires full checksum + cache.
    # Reauthenticate its actual inputs after the full restore before using it.
    dataset_root=guard.root/"datasets/transcg"
    for relative,expected in receipt["source_files"].items():
        source=guard.path(dataset_root/relative)
        require(source.is_relative_to(dataset_root) and source.stat().st_size==expected["bytes"]
                and sha256_file(source)==expected["sha256"],"Qualified sample bytes changed")
    return receipt
