"""Registered finite experiment queue. Each GPU worker owns one new process group."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8s3_training import sha,write,verify_files
from agent.m9_metrics import select_arm
from tools.m8s3_job import cleanup_group,group_members

def digest(reg):
    return hashlib.sha256(json.dumps({k:v for k,v in reg.items() if k!="registration_sha256"},
        sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()

def verify_registration(reg):
    if reg.get("registration_sha256")!=digest(reg):raise ValueError("Campaign registration changed")
    verify_files(reg["config"]["project"],reg)

def task_spec(reg,phase,arm,replicate):
    cfg=reg["config"]
    if phase not in cfg["phases"] or arm not in cfg["arms"] or replicate not in cfg["phases"][phase]["replicates"]:
        raise ValueError("Unregistered phase/arm/replicate")
    p=cfg["phases"][phase]
    task={k:cfg[k] for k in ("project","model","godot","initial_checkpoint","parent_checkpoint","warm_checkpoint","kernel_overlay")}
    task.update(cfg["common"]);task.update(cfg["arms"][arm])
    task.update(phase=phase,arm=arm,replicate=replicate,updates=p["updates"],
        train_seed_start=p["train_seed_base"]+replicate*10000,
        evaluation_seeds=list(range(p["evaluation_seed_base"]+replicate*1000,p["evaluation_seed_base"]+replicate*1000+p["evaluation_worlds"])),
        rng_seed=202611110+replicate,rng_prefix=cfg["protocol"]+"_paired",
        output=str(Path(cfg["output"])/phase/(arm+"_r"+str(replicate))),registration_sha256=reg["registration_sha256"])
    if phase=="smoke":task.update(envs=4,rollout_steps=16,minibatch=64,critic_minibatch=64,critic_epochs=2,horizon=64)
    if phase=="calibration":task.update(horizon=64)
    entry=reg.get("resume",{}).get("runs",{}).get(arm+"_r"+str(replicate)) if phase=="explore" else None
    if entry:
        task.update(resume_kind=entry["kind"],resume_source=entry["source"],resume_files=entry["files"],
            resume_registration_sha256=entry["original_registration_sha256"])
    return task

def validate_task(reg,task):
    expected=task_spec(reg,task["phase"],task["arm"],task["replicate"])
    if task!=expected:raise ValueError("Task configuration differs from pre-registered plan")

def register(config_path,destination):
    c=json.loads(Path(config_path).read_text())
    parent=json.loads(Path(c["parent_registration"]).read_text())
    source={}
    project=Path(c["project"])
    for directory in ("agent","scripts","godot_tests","tools","tests","configs"):
        for p in sorted((project/directory).rglob("*")):
            if p.is_file() and p.suffix in (".py",".gd",".json",".ps1") and "__pycache__" not in p.parts:
                source[p.relative_to(project).as_posix()]=sha(p)
    source["project.godot"]=sha(project/"project.godot")
    source["docs/M9_RESEARCH_PROTOCOL.md"]=sha(project/"docs/M9_RESEARCH_PROTOCOL.md")
    if (project/"docs/M9_RESEARCH_RESUME5.md").exists():
        source["docs/M9_RESEARCH_RESUME5.md"]=sha(project/"docs/M9_RESEARCH_RESUME5.md")
    inputs=dict(parent["input_files"])
    checkpoint=Path(c["warm_checkpoint"]);manifest=json.loads((checkpoint/"manifest.json").read_text())
    inputs["m9_warm_manifest"]=dict(path=str(checkpoint/"manifest.json"),sha256=sha(checkpoint/"manifest.json"))
    for name,h in manifest["files"].items():inputs["m9_warm:"+name]=dict(path=str(checkpoint/name),sha256=h)
    reg=dict(config=c,source_hashes=source,input_files=inputs,test_seeds=c["test_seeds"],
        parent_registration_sha256=parent["registration_sha256"],created_utc=datetime.now(timezone.utc).isoformat(),
        warm_actor_sha256=manifest["actor_sha256"],milestone_achieved=False)
    if c.get("resume_from"):
        from tools.m9_resume import collect_resume
        reg["resume"],reused_inputs=collect_resume(c)
        reg["input_files"].update(reused_inputs)
    reg["registration_sha256"]=digest(reg);verify_registration(reg)
    p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:json.dump(reg,f,ensure_ascii=False,indent=2)
    return reg

def run_owned(command,cwd,prefix,env,deadline,stopping):
    """Report actual exit code and reap all descendants even on parent failure."""
    prefix=Path(prefix)
    with prefix.with_suffix(".log").open("xb",buffering=0) as stream:
        process=subprocess.Popen(command,cwd=str(cwd),env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=stream,start_new_session=True)
        write(prefix.with_suffix(".json"),dict(pid=process.pid,process_group=process.pid,command=command,
            started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            while process.poll() is None:
                if stopping() or time.monotonic()>deadline:raise TimeoutError("Registered wall time or stop reached")
                time.sleep(.1)
        finally:
            cleanup_group(process)
    return process.returncode

def worker(reg,path,job,mode):
    if sha(path)!=json.loads((job/"launch.json").read_text())["registration_file_sha256"]:
        raise ValueError("Registration changed after submission")
    c=reg["config"];root=Path(c["output"]);root.mkdir(parents=True,exist_ok=True)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    begin=time.monotonic();deadline=begin+(c["hard_wall_seconds"] if mode=="research" else c["smoke_wall_seconds"])
    env={**os.environ,"PYTHONPATH":c["kernel_overlay"]+":"+c["project"],"HF_HUB_OFFLINE":"1",
        "TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","PYTHONUNBUFFERED":"1","CUDA_VISIBLE_DEVICES":"0"}
    stages=[];comparisons=[];status="failed";error=None
    def stage(name,command):
        write(job/"status.json",dict(status="running",stage=name,elapsed_seconds=time.monotonic()-begin,
            remaining_hard_wall_seconds=max(0,deadline-time.monotonic()),milestone_achieved=False))
        code=run_owned([sys.executable,"-u",*command],c["project"],job/name,env,deadline,lambda:stopped)
        stages.append(dict(stage=name,exit_code=code));write(job/"stages.json",stages)
        if code:raise RuntimeError(name+" exited "+str(code))
    def run_task(phase,arm,rep):
        spec=task_spec(reg,phase,arm,rep)
        cfg=job/(phase+"_"+arm+"_r"+str(rep)+"_config.json")
        if cfg.exists():raise FileExistsError("Task already exists")
        write(cfg,spec)
        label=phase+"_"+arm+"_r"+str(rep)
        command=["-m","agent.m9_run","--campaign",str(path),"--config",str(cfg)]
        if spec.get("resume_kind")=="complete":
            from tools.m9_resume import import_completed
            import_completed(spec)
            write(job/(label+"_import.json"),dict(source=spec["resume_source"],kind="immutable_result_import",gpu_worker_started=False))
            return spec,cfg,command
        stage(label,command+(["--action","recover"] if spec.get("resume_kind")=="evaluate_saved" else []))
        stage(label+"_reload",command+["--action","reload"])
        return spec,cfg,command
    try:
        verify_registration(reg)
        arms=list(c["arms"])
        if mode=="calibration":
            spec,cfg,command=run_task("calibration","control",0)
            stage("calibration_control_audit",["tools/audit_m9.py","--campaign",str(path),"--config",str(cfg)])
            status="calibration_completed"
        elif mode=="smoke":
            for arm in arms:
                spec,cfg,command=run_task("smoke",arm,0)
                stage("smoke_"+arm+"_audit",["tools/audit_m9.py","--campaign",str(path),"--config",str(cfg)])
                data=json.loads((Path(spec["output"])/"training_complete.json").read_text())
                comparisons.append(dict(arm=arm,replicate=0,**data))
                write(job/"smoke_results.json",comparisons)
            status="smoke_completed"
        else:
            for rep in c["phases"]["explore"]["replicates"]:
                for arm in arms:
                    spec,cfg,command=run_task("explore",arm,rep)
                    data=json.loads((Path(spec["output"])/"evaluation.json").read_text())
                    initial=json.loads((Path(spec["output"])/"initial.json").read_text())
                    comparisons.append(dict(arm=arm,replicate=rep,summary=data["summary"],initial=initial["summary"],
                        output=spec["output"],actor_sha256=data["actor_sha256"]))
                    write(root/"comparisons.json",comparisons)
            ranking=select_arm(comparisons,arms)
            write(root/"explore_ranking.json",dict(ranking=ranking,selected=ranking[:2],registration_sha256=reg["registration_sha256"]))
            confirmations=[]
            for arm in ranking[:2]:
                spec,cfg,command=run_task("confirm",arm,2)
                stage("confirm_"+arm+"_audit",["tools/audit_m9.py","--campaign",str(path),"--config",str(cfg)])
                data=json.loads((Path(spec["output"])/"evaluation.json").read_text())
                confirmations.append(dict(arm=arm,summary=data["summary"],config=str(cfg),output=spec["output"],command=command))
            winner=max(confirmations,key=lambda r:(r["summary"]["successes"],r["summary"]["cvar20"],r["summary"]["rmst"],-arms.index(r["arm"])))
            eligible=winner["summary"]["successes"]>=23 and winner["summary"]["successes_10day"]>=23 and winner["summary"]["early_deaths"]==0
            write(root/"selection.json",dict(winner=winner,confirmations=confirmations,eligible_for_test=eligible,milestone_achieved=False))
            if eligible:
                stage("final_test",winner["command"]+["--action","test"])
                stage("final_test_audit",["tools/audit_m9.py","--campaign",str(path),"--config",winner["config"],"--evaluation-only"])
                gate=json.loads((Path(winner["output"])/"numerical_gate.json").read_text())
                status="numerical_pass_pending_online_acceptance" if gate["numerical_gate_passed"] else "numerical_gate_failed"
            else:status="research_completed_selection_failed"
        verify_registration(reg)
    except BaseException as exc:
        error=type(exc).__name__+": "+str(exc)
        raise
    finally:
        gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader"],
            capture_output=True,text=True,timeout=10)
        remaining={}
        for entry in job.glob("*.json"):
            item=json.loads(entry.read_text())
            if isinstance(item,dict) and "process_group" in item:
                members=group_members(item["process_group"])
                if members:remaining[str(item["process_group"])]=members
        result=dict(status=status,error=error,seconds=time.monotonic()-begin,stages=stages,
            finished_utc=datetime.now(timezone.utc).isoformat(),remaining_gpu_compute_processes=gpu.stdout.strip(),
            gpu_query_exit_code=gpu.returncode,owned_worker_groups_cleaned=not remaining,
            remaining_owned_processes=remaining,milestone_achieved=False)
        write(job/"finished.json",result);write(job/"status.json",result)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["register","start","_run"])
    ap.add_argument("--config");ap.add_argument("--registration",required=True);ap.add_argument("--job")
    ap.add_argument("--mode",choices=["smoke","calibration","research"],default="research")
    a=ap.parse_args();path=Path(a.registration).resolve()
    if a.action=="register":
        reg=register(a.config,path);print(json.dumps(dict(registration_sha256=reg["registration_sha256"],sources=len(reg["source_hashes"]),inputs=len(reg["input_files"]))))
        return
    reg=json.loads(path.read_text());verify_registration(reg);job=Path(a.job).resolve()
    if a.action=="_run":worker(reg,path,job,a.mode);return
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip()
    if active:raise RuntimeError("GPU is busy: "+active)
    job.mkdir(parents=True,exist_ok=False)
    write(job/"launch.json",dict(registration=str(path),registration_file_sha256=sha(path),mode=a.mode,created_utc=datetime.now(timezone.utc).isoformat()))
    with (job/"manager.log").open("xb",buffering=0) as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"_run","--registration",str(path),
            "--job",str(job),"--mode",a.mode],cwd=reg["config"]["project"],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=stream,start_new_session=True)
    write(job/"manager.json",dict(pid=p.pid,registration_sha256=reg["registration_sha256"]))
    print(json.dumps(dict(pid=p.pid,job=str(job),registration_sha256=reg["registration_sha256"])))
if __name__=="__main__":main()

