from pathlib import Path
import hashlib,json,re
from check_acceptance import validate
ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
run=ROOT/'evidence/run05'
assert validate(read(run/'report.json'),read(run/'independent_reload.json'))
assert read(ROOT/'download_verification.json')['all_passed']
documents=[ROOT/'report.md',ROOT/'final_review.md',WORKSPACE/'docs/MEA_LLM_OVERALL_PLAN.md',WORKSPACE/'docs/superpowers/plans/2026-11-16-mea-22554-gpu-smoke.md']
links=0
for doc in documents:
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):continue
        assert (doc.parent/target.split('#')[0]).resolve().exists(),(doc,target)
        links+=1
counts={f'run{i:02d}':len((ROOT/f'evidence/run{i:02d}/rollout.jsonl').read_text(encoding='utf-8').splitlines()) for i in range(2,6)}
assert set(counts.values())=={32} and sum(counts.values())==128
assert read(ROOT/'evidence/run05_exit.json')['exit_code']==0
result=dict(accepted_run='run05',local_reload_training_interface_passed=True,cross_host_probability_reproduction_passed=False,
    r2_enabled=False,stimulation_performed=False,accepted_decisions=32,all_debug_decisions=128,
    reviewed_critical=0,reviewed_important=0,reviewed_minor=0,local_links_checked=links,
    task_files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='final_delivery.json'},
    documents={str(p.relative_to(WORKSPACE)):sha(p) for p in documents})
with (ROOT/'final_delivery.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['task_files','documents']},indent=2))
