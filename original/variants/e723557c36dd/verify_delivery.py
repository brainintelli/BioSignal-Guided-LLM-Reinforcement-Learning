from pathlib import Path
import hashlib,json,tarfile,datetime
B=Path(__file__).resolve().parent;p=B/'closed_session_evidence.tar.gz'
digest=hashlib.sha256(p.read_bytes()).hexdigest()
assert digest=='804be52f4282d81dec092ff46be2f3b2761dd0f00bc4d646fa318481795bb41c'
assert p.stat().st_size==1256814
dest=B/'evidence';dest.mkdir(exist_ok=False)
with tarfile.open(p,'r:gz') as t:t.extractall(dest,filter='data')
manifest=json.loads((dest/'evidence_manifest.json').read_text())
for row in manifest:
 f=dest/row['path'];assert f.stat().st_size==row['bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],row['path']
reg=json.loads((dest/'analysis_registration.json').read_text())
for name,digest in reg['source_sha256'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,name
state=json.loads((dest/'state.json').read_text());checks=json.loads((dest/'delivery_checks.json').read_text())
assert state['status']=='stopped' and not state['cleanup_errors']
assert set(state['stage_files'])=={'baseline','pilot'} and not (dest/'gate_pilot.json').exists()
assert checks['cumulative_reserved_true_sham']==[6,1] and checks['cumulative_biological_seconds']<7200
q=json.loads((dest/'pilot_qc.json').read_text());assert q['status']=='failed' and q['event_ids']==[8201,8202]
summary=json.loads((dest/'analysis/pilot_summary.json').read_text());assert summary['status']=='qc_failed_diagnostic_only'
assert summary['h5_sha256']==next(x['sha256'] for x in checks['source_h5'] if x['stage']=='pilot')
r={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'manifest_files_verified':len(manifest),'registered_source_hashes_verified':len(reg['source_sha256']),
 'formal_control_not_run':True,'pilot_qc_failed_preserved':True,'closed_session_checks':checks}
(B/'local_delivery_verification.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
