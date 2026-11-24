"""Locally accept hashed, independently replayed results before drawing efficacy plots."""
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import sys
import zipfile

WORK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(WORK/'standalone/mea_state2_r2'),str(WORK/'standalone/mea_binary_r2'),str(WORK)]
from mea_state2_r2.features import digest
from mea_state2_r2.analysis import analyze

EXPECTED='59cdfd3a99063d94fbc720b4340ad12245bd482429dc9b72a8ab9c6f60231e12'
SOURCE='5283b5f538b85b4f89e28b99ccf844cd24cb39ce547c70fdeead71cd66ea9094'
ACTOR='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise ValueError(message)


def derived(rows,horizon):
    rows=rows[:horizon]
    require(bool(rows),'Empty trajectory')
    last=rows[-1]['transition']['after']
    r1=math.fsum(r['r1_score']['reward'] for r in rows)
    ticks=sum(r['duration'] for r in rows)
    return dict(r1_sum=r1,r1_per_tick=r1/ticks,visited_count=last['visited_count'],
        final_health=last['health'],final_hunger=last['hunger'],final_thirst=last['thirst'],
        elapsed_ticks=ticks,terminated=int(rows[-1]['terminal']),decisions=len(rows),
        decision_budget=horizon,absorbed_slots=horizon-len(rows) if rows[-1]['terminal'] else 0)


def accept(root):
    registration=read(root/'registration.json')
    require(digest(registration)==EXPECTED,'Wrong registration')
    require(file_sha(root/'manifest.json')==SOURCE,'Wrong source manifest')
    require(registration['source_manifest_sha256']==SOURCE,'Registration/source disagreement')
    manifest=read(root/'evidence_manifest.json')
    for name,spec in manifest.items():
        p=(root/name).resolve()
        require(p.is_relative_to(root.resolve()),'Escaped manifest path')
        require(p.stat().st_size==spec['size'] and file_sha(p)==spec['sha256'],'Evidence changed: '+name)
    status=read(root/'campaign_status.json')
    require(status['complete'] is True and status['passed'] is True and status['accepted_cells']==40,'Campaign incomplete')
    require(status['registration_sha256']==EXPECTED,'Campaign registration mismatch')
    require(read(root/'preflight.json')['passed'] is True,'Preflight not passed')
    cells=[]
    for block in range(8):
        for arm in ('A0','A','B','C','D'):
            cellroot=root/'experiment01'/f'b{block:02d}_{arm}'
            cell=read(cellroot/'cell_summary.json');state=read(cellroot/'status.json')
            verified=read(cellroot/'verification/verified.json');evaluation=read(cellroot/'eval/evaluation.json')
            for value in (cell,state):
                require(value['registration_sha256']==EXPECTED and value['passed'] is True,'Unaccepted cell')
            for value in (cell,verified,evaluation):
                require(value['block']==block and value['arm']==arm,'Wrong cell identity')
            require(cell['smoke'] is False and verified['smoke'] is False,'Smoke result in formal campaign')
            require(verified['passed'] is True and verified['hardware_commands_sent']==0 and verified['real_output_verified'] is False,'Invalid verification')
            require(cell['verification_sha256']==file_sha(cellroot/'verification/verified.json'),'Verification receipt changed')
            require(cell['evaluation_sha256']==file_sha(cellroot/'eval/evaluation.json'),'Evaluation receipt changed')
            require(evaluation['initial']['actor']==ACTOR,'Wrong initial actor')
            require(cell['metrics']==evaluation['metrics'] and cell['evaluated']==evaluation['evaluated'],'Evaluation mirror mismatch')
            require(all(s['exit_code']==0 for s in state['stages']),'Failed process stage')
            for phase in ('after','before'):
                v=verified['evaluation'][phase]
                rows=[json.loads(s) for s in (cellroot/'eval'/phase/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
                require(len(rows)==150 or (0<len(rows)<150 and rows[-1]['terminal']),'Incomplete rollout')
                require(v['passed'] is True and v['decisions']==len(rows),'Replay not passed')
                require(all(r['index']==i and r['action_source']=='model' for i,r in enumerate(rows)),'Action provenance mismatch')
                for horizon,n in [('full',150),('first16',16)]:
                    m=derived(rows,n)
                    require(m==v['metrics'][horizon]==cell['metrics'][phase][horizon],'Local outcome aggregation differs')
            if arm=='A0':
                require(verified['training_decisions']==0 and cell['evaluated']==evaluation['initial'],'Frozen baseline changed')
            else:
                require(verified['training_decisions']==384,'Training budget differs')
                require(cell['training']==verified['training_summary'],'Training mirror mismatch')
                require(cell['training']['terminals']==0,
                    'Frozen v1 terminal-context bug: training deaths require a new complete corrected campaign')
                training=read(cellroot/'train/training_result.json')
                require(training['final']==cell['evaluated'],'Wrong final checkpoint evaluated')
                require(training['hardware_commands_sent']==0 and training['r2_source']=='simulation_from_game_state','Wrong reward provenance')
                require(sum(r['decisions'] for r in training['rounds'])==384,'Round budget differs')
                require(all(all(r['checks'].values()) for r in training['rounds']),'Training checks failed')
            cells.append(cell)
    analysis=analyze(cells,EXPECTED)
    require(analysis['status']=='complete','Analysis incomplete')
    return {'registration_sha256':EXPECTED,'cells':cells,'local_audit':{'passed':True,'cells':len(cells),
        'hashed_files':len(manifest),'training_decisions':32*384,'evaluation_scenarios':80,
        'hardware_commands_sent':0,'digital_only':True}}


def main():
    archive=Path(sys.argv[1]);receipt=read(Path(sys.argv[2]));out=archive.parent/'accepted_evidence'
    require(file_sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['size'],'Export archive hash mismatch')
    out.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(set(names))==len(names),'Duplicate ZIP entry')
        for name in names:
            posix=PurePosixPath(name)
            require(not posix.is_absolute() and '..' not in posix.parts and '\\' not in name and ':' not in name,'Unsafe ZIP path')
            require((out/name).resolve().is_relative_to(out.resolve()),'Escaped ZIP path')
        z.extractall(out)
    accepted=accept(out)
    with (archive.parent/'accepted_results.json').open('x',encoding='utf-8') as f:
        json.dump(accepted,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(accepted['local_audit']))


if __name__=='__main__':main()
