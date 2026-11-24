"""Export accepted text evidence; model weights stay on the audited server."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def export(root,output):
    root=root.resolve();exp=root/'experiment01'
    audit=json.loads((root/'independent_audit.json').read_text())
    status=json.loads((exp/'parallel_status.json').read_text())
    job=json.loads((root/'job.json').read_text())
    if (audit.get('passed') is not True or audit.get('cells')!=32 or status.get('passed') is not True
            or status.get('accepted_cells')!=32 or job.get('accepted_complete') is not True):
        raise ValueError('Only fully accepted, independently audited experiment may be exported')
    names=['independent_audit.json','job.json','launch.json','gpu_preflight.json','gpu_tests_receipt.json',
           'memory_probe_receipt.json','preflight_job.json','project_source_manifest.json',
           'project_frozen.zip','model_config.json','autonomous_runtime_recheck.json',
           'technical_amendment.json','design_pin.json','independent_audit.py']
    selected=[root/name for name in names]
    omitted=[]
    for path in sorted(exp.rglob('*')):
        if not path.is_file():continue
        if path.suffix in ('.json','.jsonl','.log'):selected.append(path)
        else:omitted.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size))
    selected.extend(sorted((root/'data').glob('*.json')))
    included={}
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Unexpected external source')
        name=path.relative_to(root).as_posix()
        if name in included:raise ValueError('Duplicate export entry')
        included[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    index=dict(schema='mea.evidence.subset.v1',complete_experiment_audited=True,
        full_model_weights_downloaded=False,remote_root=str(root),included=included,
        omitted_binary_files=omitted,omitted_hashes='See audited per-cell cell_manifest.json for each omitted file',
        input_audit_sha256=sha(root/'independent_audit.json'))
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in selected:archive.write(path,path.relative_to(root).as_posix())
        archive.writestr('__bundle__/manifest.json',json.dumps(index,indent=2))
    receipt=dict(passed=True,archive=str(output),archive_sha256=sha(output),archive_bytes=output.stat().st_size,
                 included_files=len(included),omitted_binary_files=len(omitted),
                 omitted_binary_bytes=sum(row['bytes'] for row in omitted),full_model_weights_downloaded=False)
    with output.with_suffix('.receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();export(args.root,args.output)
