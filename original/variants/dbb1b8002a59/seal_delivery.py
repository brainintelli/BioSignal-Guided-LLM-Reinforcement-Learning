"""Run on GPU host only AFTER all registered cells and analysis complete."""
from pathlib import Path
from datetime import datetime,timezone
import itertools
import json
import sys
import zipfile

ROOT=Path('/home/researcher/mea-long-ablation-20261116')
PROJECT=ROOT/'project'
SOURCE=Path('/home/researcher/mea-long-20261116/long_28080_admitted_20261116')
GODOT=Path('/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64')
sys.path.insert(0,str(PROJECT))
from agent.mea_long_ablation import ARMS,PIN,analyze
from agent.mea_protocol import digest
from tools.mea_long_ablation import registration,validate_receipt
from tools.mea_long_train import check_manifest
from tools.mea_p0 import read_json,write_json,sha,_same


def main():
    campaign=ROOT/'experiment01'
    status=read_json(campaign/'batch_status.json');result=read_json(campaign/'analysis.json')
    if status.get('complete') is not True or status.get('passed') is not True or result.get('status')!='complete':
        raise ValueError('campaign not fully accepted')
    planned=set(itertools.product(range(8),ARMS))
    if len(status['cells'])!=24 or {(c['block'],c['arm']) for c in status['cells']}!=planned:
        raise ValueError('cell inventory differs')
    for c in status['cells']:
        if [s['stage'] for s in c['stages']]!=['train','evaluate','verify'] or any(s['exit_code']!=0 for s in c['stages']):
            raise ValueError('failed or incomplete stage')
    final_line=(ROOT/'batch.log').read_text().splitlines()[-1]
    _same(json.loads(final_line),dict(batch_passed=True,analysis_status='complete',positive_evidence=result['positive_evidence']),'batch final receipt')
    reg,data=registration(campaign,SOURCE,ROOT/'model_config.json',PROJECT,GODOT)
    _same(result['registration_sha256'],digest(reg),'analysis registration')
    _same(result['source_manifest_sha256'],PIN,'analysis source')
    cells=[]
    for b,arm in sorted(planned):
        out=campaign/f'b{b:02d}_{arm}';receipt=read_json(campaign/f'verified_b{b:02d}_{arm}.json')
        validate_receipt(receipt,b,arm,digest(reg),sha(out/'cell_manifest.json'))
        check_manifest(out,'cell_manifest.json')
        _same(sha(out/'train/independent_verification.json'),receipt['training_verification_sha256'],'training receipt')
        evaluation=read_json(out/'evaluation.json');training=read_json(out/'train/report.json')
        _same(training['initial'],reg['common_initial'],'initial tensors')
        _same(training['decisions'],1024,'training count')
        for r in training['rounds']:_same(r['applied_steps'],{'actor':2,'critic':16},'budget')
        cells.append(dict(block=b,arm=arm,**{p:evaluation['metrics'][p]['discounted_r1'] for p in ('pre','post')}))
    for k,v in analyze(cells).items():_same(result[k],v,'recomputed analysis '+k)
    files={}
    for path in sorted(campaign.rglob('*')):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(campaign.resolve()):raise ValueError('escaped path')
            files[path.relative_to(campaign).as_posix()]={'size':path.stat().st_size,'sha256':sha(path)}
    manifest=dict(schema='mea.long.ablation.delivery.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=digest(reg),source_manifest_sha256=PIN,accepted_cells=24,files=files)
    write_json(ROOT/'accepted_campaign_manifest.json',manifest)
    extras=['accepted_campaign_manifest.json','project_frozen.zip','project_source_manifest.json','model_config.json',
        'preflight_tests_final.log','admission_verify.log','registration.log','launch.json','batch.log','seal_delivery.py']
    archive=ROOT/'campaign_delivery.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in files:z.write(campaign/name,'experiment01/'+name)
        for name in extras:z.write(ROOT/name,name)
        z.write(SOURCE/'manifest.json','source_manifest.json')
    receipt=dict(passed=True,verified_utc=datetime.now(timezone.utc).isoformat(),accepted_cells=24,
        archived_campaign_files=len(files),archive_size=archive.stat().st_size,archive_sha256=sha(archive),
        accepted_manifest_sha256=sha(ROOT/'accepted_campaign_manifest.json'),source_manifest_sha256=PIN,
        full_checkpoints_archived=True,source=data.summary())
    write_json(ROOT/'archive_receipt.json',receipt)
    print('ARCHIVE_BYTES',receipt['archive_size'],flush=True)
    for label,key in [('ARCHIVE','archive_sha256'),('MANIFEST','accepted_manifest_sha256')]:
        print(label+'_SHA1',receipt[key][:32],flush=True);print(label+'_SHA2',receipt[key][32:],flush=True)
    print('ARCHIVED_FILES',len(files),flush=True)


if __name__=='__main__':main()
