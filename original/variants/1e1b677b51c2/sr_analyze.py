"""Accept only complete, independently verified cells; evaluate original R1."""
import argparse,json,math
from pathlib import Path
from sr_data import read,write,sha
from sr_protocol import ARMS,PROTOCOL,analyze,episode_metrics,reward_record

def validate_receipt(report,v,reg_sha,manifest_sha):
    for k,x in dict(passed=True,block=report['block'],arm=report['arm'],collection_pid=report['collection_pid'],registration_sha256=reg_sha,
        collection_manifest_sha256=manifest_sha,actual_weight_optimizer_checks=True).items():
        if type(v.get(k)) is not type(x) or v[k]!=x:raise ValueError('verification receipt binding '+k)
    if v['verification_pid']==report['collection_pid']:raise ValueError('same process verification')
    if [x['phase'] for x in v['phases']]!=['pre0','pre1','train','post0','post1']:raise ValueError('verification phase inventory')
    for p in v['phases']:
        if any(p.get(k) is not True for k in ['native_replay','snapshot_inputs_verified','reward_gae_verified']):raise ValueError('incomplete replay')
    errors=[v['updated_train_statistic_error']]+[p[k] for p in v['phases'] for k in ('statistic_error','bootstrap_error')]
    if any(not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1e-6 for x in errors):raise ValueError('nonfinite/failed statistics')
    return True

def gather(root):
    reg=read(root/'registration.json');pin=read(root/'registration_pin.json')['sha256'];execution=read(root/'execution_plan.json')
    if reg['pilot'] or reg['protocol']!=PROTOCOL or sha(root/'registration.json')!=pin or execution['registration_sha256']!=pin:raise ValueError('registration identity')
    if execution['code']['sr_analyze.py']!=sha(Path(__file__)):raise ValueError('analysis code changed after outcomes')
    cells=[]
    for block in range(8):
        for arm in ARMS:
            out=root/f'b{block:02d}_{arm}'
            if not (out/'independent_verification.json').exists():continue
            manifest=read(out/'collection_manifest.json');actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}-{'collection_manifest.json','independent_verification.json'}
            if set(manifest)!=actual:raise ValueError('cell inventory changed')
            for n,h in manifest.items():
                p=(out/n).resolve()
                if not p.is_relative_to(out.resolve()) or sha(p)!=h:raise ValueError('cell byte hash mismatch '+n)
            report=read(out/'report.json');v=read(out/'independent_verification.json')
            if report['passed'] is not True or report['pilot'] or report['block']!=block or report['arm']!=arm or report['registration_sha256']!=pin:raise ValueError('collection binding')
            validate_receipt(report,v,pin,sha(out/'collection_manifest.json'))
            identity=read(out/'identity.json')
            if identity['initial']!=reg['common_initial'] or identity['source']!=reg['source_identity'] or identity['config']!=reg['configs'][f'{block}/{arm}']:raise ValueError('common start/source/config')
            if report['gpu']['physical_gpu']!=block or v['gpu']['uuid']!=report['gpu']['uuid']:raise ValueError('paired GPU mismatch')
            for j,name in enumerate(['pre0','pre1','train','post0','post1']):
                phase=out/name;events=[json.loads(s) for s in (phase/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
                metrics=read(phase/'metrics.json');rows=[]
                for e in events:
                    r=e['training'];rr=reward_record(e['r1_score']['reward'],e['snapshot']['score'],arm,training=name=='train')
                    if e['reward_record']!=rr or r['reward']!=rr['total_reward'] or r['duration']!=e['r1_score']['elapsed_ticks']:raise ValueError('R1/aux decomposition')
                    rows.append(dict(r1=rr['r1'],duration=r['duration'],terminal=r['terminal']))
                derived=episode_metrics(rows)
                if any(metrics[k]!=x for k,x in derived.items()) or v['phases'][j]['decisions']!=len(rows):raise ValueError('recomputed R1 mismatch')
                expected=report['train'] if name=='train' else report['pre' if name.startswith('pre') else 'post'][int(name[-1])]
                if metrics!=expected:raise ValueError('report metrics mismatch')
            cells.append(dict(block=block,arm=arm,pre=report['pre'],post=report['post'],verified=True))
    return cells

def main(root):
    cells=gather(root);result=analyze(cells)
    if result['status']=='complete' and read(root/'driver_status.json')['passed'] is not True:raise ValueError('driver did not finish cleanly')
    write(root/'analysis.json',result);print(json.dumps(dict(status=result['status'],verified_cells=len(cells),primary=result.get('primary')),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
