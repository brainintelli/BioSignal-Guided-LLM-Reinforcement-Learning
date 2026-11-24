"""Check delivered evidence and local document links without modifying prior results."""
from pathlib import Path
import hashlib,json,re

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
PREVIOUS=ROOT.parent/'mea_22554_baseline_20261116'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
transfer=ROOT/'model_transfer'
expected={'bundle_manifest.json':'d18171bae74bf48581ae3572bfe47cda'+'f3b73fa046e3a3478ea5b3af7f1f0517',
          'bundle_summary.json':'44dd5738a9ea180790de4bde1b2dbd6d'+'572b9236f53128b7ce8da1341990daf9'}
for name,h in expected.items(): assert sha(transfer/name)==h
manifest=json.loads((transfer/'bundle_manifest.json').read_text())
summary=json.loads((transfer/'bundle_summary.json').read_text())
verification=json.loads((transfer/'verification.json').read_text())
assert len(manifest['files'])==verification['verified_files']==384
assert sum(f['size'] for f in manifest['files'].values())==verification['verified_bytes']==2039667227
assert verification['archive_sha256']==summary['archive_sha256']
assert verification['manifest_sha256']==sha(transfer/'bundle_manifest.json')
assert verification['all_file_hashes_match'] and not verification['training_started']
assert not verification['runtime_tested_on_destination']
checkpoint=summary['target_checkpoint']
assert verification['adapter_file_sha256']==manifest['files'][checkpoint+'/adapter/adapter_model.safetensors']['sha256']
assert verification['recorded_actor_sha256']==summary['actor_sha256']
for name,entry in manifest['files'].items():
    assert len(entry['sha256'])==64 and all(c in '0123456789abcdef' for c in entry['sha256'])
previous=json.loads((PREVIOUS/'delivery_manifest.json').read_text())
for name,h in previous['outputs'].items(): assert sha(PREVIOUS/name)==h,name
identity=json.loads((ROOT/'stress_v1/identity.json').read_text())
for name,path in [('script',ROOT/'stress_candidates.py'),('protocol',ROOT/'analysis_protocol.md'),('analysis',ROOT/'stress_v1/analysis.json'),('trials',ROOT/'stress_v1/all_trials.csv')]:
    assert sha(path)==identity[name+'_sha256']
independent=json.loads((ROOT/'stress_v1/independent_verification.json').read_text())
assert independent['all_checks_passed'] and independent['verified_rows']==7028
assert independent['verifier_sha256']==sha(ROOT/'verify_and_plot_stress.py')
documents=[ROOT/'report.md',WORKSPACE/'docs/MEA_LLM_OVERALL_PLAN.md',WORKSPACE/'docs/superpowers/plans/2026-11-16-mea-22554-baseline-r2.md']
links=0
for doc in documents:
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):continue
        p=(doc.parent/target.split('#')[0]).resolve()
        assert p.exists(),(doc,target)
        links+=1
result=dict(schema='mea.followup.delivery.v1',previous_delivery_hashes_unchanged=len(previous['outputs']),
    source_transfer_metadata_hashes_match=True,model_files_verified_remotely=384,score_rows_verified=7028,
    local_document_links_checked=links,training_started=False,r2_enabled=False,
    outputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
    documents={str(p.relative_to(WORKSPACE)):sha(p) for p in documents})
with (ROOT/'delivery_manifest.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','documents')},indent=2))
