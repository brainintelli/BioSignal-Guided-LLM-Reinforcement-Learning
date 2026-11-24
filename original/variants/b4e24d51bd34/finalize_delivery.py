"""Index final human-facing artifacts without changing frozen experiment files."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;PROJECT=R.parent.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    verification=json.loads((R/'download_verification.json').read_text(encoding='utf-8'))
    assert verification['passed'] and (R/'final_review.md').exists()
    paths=[p for p in R.iterdir() if p.is_file() and p.suffix in {'.py','.md','.json','.log'} and p.name!='delivery_index.json']
    paths += [PROJECT/'docs/superpowers/specs/2026-11-16-mea-22554-state-reward-design.md',PROJECT/'docs/superpowers/plans/2026-11-16-mea-22554-state-reward.md',PROJECT/'docs/MEA_LLM_OVERALL_PLAN.md',R/'evidence/evidence_manifest.json']
    result=dict(complete=True,verified_evidence_files=verification['files'],registered_cells=40,
        archive_sha256=json.loads((R/'transport.json').read_text())['sha256'],
        artifacts={p.relative_to(PROJECT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths})
    with (R/'delivery_index.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(complete=True,indexed_artifacts=len(paths),verified_evidence_files=verification['files'])))
if __name__=='__main__':main()
