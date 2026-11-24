"""GPU-only RNG reproduction from recorded probabilities; never load an actor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8b_eval import episode_seed,DECODING
from agent.m8b_identity import runtime_identity
from agent.m8b_registration import verify_registration
from tools.rl_audit_io import iter_trace_lines,file_sha256


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--registration",type=Path,required=True);p.add_argument("--report",type=Path,required=True)
    p.add_argument("--trace",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    registration=json.loads(a.registration.read_text(encoding='utf-8'))
    config=registration['config'];verify_registration(config['project'],registration)
    report=json.loads(a.report.read_text(encoding='utf-8'))
    runtime=registration['execution_identity']['runtime']
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; CPU RNG is not equivalent')
    if runtime_identity(torch)!=runtime:raise RuntimeError('Recorded GPU/runtime identity differs')
    if a.output.exists():raise FileExistsError('Preserve previous RNG audit')
    if file_sha256(a.trace)!=report['trace_sha256']:raise ValueError('Report/trace SHA differs')
    seeds=report['seeds'];replicates=report['replicates'];arm=report['arm'];profile='open1' if arm=='original_v2' else 'open2'
    from tools.audit_m8b_run import validate_environment_identity
    validate_environment_identity(report,registration)
    if not ((seeds==config['validation_seeds'] and replicates==config['replicates']) or
            (seeds==config['test_seeds'] and replicates==config['test_replicates'])):
        raise ValueError('Episode matrix is not registered')
    if report['source_sha256']!=registration['source_hashes'] or report['decoding']!=DECODING or report['temperature']!=1.:
        raise ValueError('Report identity/decoding differs')
    expected={(s,r) for s in seeds for r in replicates}
    contexts={};count=0
    def digest(generator):return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    for line in iter_trace_lines(a.trace):
        if not line.strip():continue
        row=json.loads(line)
        validate_environment_identity(row,registration,arm=arm,profile=row['observation']['profile'])
        key=(row["seed"],row["replicate"])
        if row.get('actor_sha256')!=report['actor_sha256'] or row.get('contract')!=report['contract'] or row.get('temperature')!=1.:
            raise AssertionError('Row actor/contract/temperature differs from report')
        if row["observation"]["profile"]!=profile:raise AssertionError("Wrong terrain profile")
        if key not in expected or row["rng_seed"]!=episode_seed(config["rng_prefix"],profile,*key) or row["decoding"]!=DECODING:
            raise AssertionError("World/replicate/RNG/decoding identity mismatch")
        if key not in contexts:
            if row["observation"]["tick"]!=0:raise AssertionError("Episode does not start at zero")
            contexts[key]=dict(generator=torch.Generator(device="cuda").manual_seed(row["rng_seed"]),ended=False)
        c=contexts[key]
        if c["ended"]:raise AssertionError("Episode continued after terminal")
        g=c["generator"]
        if digest(g)!=row["rng_before_sha256"]:raise AssertionError("RNG before state mismatch")
        for probs,chosen in ((row["verb_probabilities"],row["verb_index"]),(row["target_probabilities"],row["target_index"])):
            probabilities=torch.tensor(probs,device="cuda",dtype=torch.float32)
            if not torch.isfinite(probabilities).all() or (probabilities<0).any() or abs(float(probabilities.sum())-1)>1.e-5:
                raise AssertionError("Invalid stored probabilities")
            sampled=int(torch.multinomial(probabilities,1,generator=g).item())
            if sampled!=chosen:raise AssertionError("Recorded action differs from exact RNG draw")
        if digest(g)!=row["rng_after_sha256"]:raise AssertionError("RNG after state mismatch")
        c["ended"]=row["terminal"] or row["truncated"];count+=1
    if set(contexts)!=expected or not all(c["ended"] for c in contexts.values()):raise AssertionError("Incomplete episode matrix")
    if count!=sum(e['decisions'] for e in report['episodes']):raise AssertionError('Report/trace decision count differs')
    result=dict(passed=True,arm=arm,profile=profile,replayed_decisions=count,episodes=len(contexts),actor_loaded=False,new_world_steps=0,
        checks="Per-episode CUDA generator; two multinomial draws; exact selected indices and before/after RNG state SHA",
        trace_sha256=file_sha256(a.trace),runtime=runtime,report_sha256=file_sha256(a.report),registration_sha256=file_sha256(a.registration),
        actor_sha256=report["actor_sha256"],rng_sampling_reproduced=True,
        scope="RNG reproduction only; does not prove stored probabilities originated from the actor. Godot/input and actor identity audits are separate")
    with a.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
