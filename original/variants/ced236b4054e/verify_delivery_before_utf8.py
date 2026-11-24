"""Local byte, source, trace, GAE, pairing and statistical delivery audit."""
import hashlib,itertools,json,math,statistics
from pathlib import Path
from independent_analysis import statistics_for
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence';EXP=E/'experiment01'
ARMS=('baseline','state','reward','both','shuffled');PIN='c06cfa4acdbafd856a4ff3929c5847657f96c68e2d38c74f9c1c6df527ef2000'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def audit_gae(rows,boot):
    carry=0.
    for i in reversed(range(len(rows))):
        r=rows[i];assert type(r['duration']) is int and r['duration']>0
        assert r['boundary']==(r['terminal'] or i==len(rows)-1)
        nv=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else boot
        discount=.99985**r['duration'];carry=r['reward']+discount*nv-r['value']+discount*.999**r['duration']*(not r['boundary'])*carry
        assert r['next_value']==nv and math.isfinite(carry) and abs(carry-r['advantage'])<1e-10 and abs(carry+r['value']-r['return'])<1e-10
def main():
    manifest=read(E/'evidence_manifest.json')
    for n,v in manifest.items():
        p=(E/n).resolve();assert p.is_relative_to(E.resolve()) and p.stat().st_size==v['size'] and sha(p)==v['sha256'],n
    for n in [p.name for p in E.glob('*.py')]:
        assert (ROOT/n).exists() and sha(ROOT/n)==sha(E/n),n
    reg=read(EXP/'registration.json');reg_sha=sha(EXP/'registration.json');launch=read(EXP/'execution_plan.json')
    assert reg_sha==read(EXP/'registration_pin.json')['sha256']==launch['registration_sha256']
    assert not reg['pilot'] and not reg['stimulation'] and reg['common_initial']['actor']=='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
    assert sha(E/'dataset.json')==sha(ROOT/'dataset.json')==PIN==reg['source_identity']['dataset_sha256']
    assert sha(ROOT.parent/'mea_22554_baseline_20261116/source/22554_baseline.raw.h5')==reg['protocol']['source_sha256']
    assert read(ROOT/'data_independent_audit.json')['passed']
    for n,h in reg['source_identity']['files'].items():assert sha(E/n)==h
    for n,h in launch['code'].items():assert sha(E/n)==h
    assert read(E/'runtime_audit.json')['passed'] and read(E/'runtime_audit.json')['files']==24551 and read(E/'runtime_audit.json')['bytes']==5823470108
    assert read(E/'post_run_inventory.json')['passed'] and read(E/'post_run_inventory.json')['owned_pids_alive']==[] and read(E/'post_run_inventory.json')['remaining_task_processes']==[] and not read(E/'post_run_inventory.json')['compute'].strip()
    assert read(EXP/'driver_status.json')['passed'] and read(EXP/'driver_status.json')['finished_cells']==40
    for n in ('pilot01_register','pilot01_collect','pilot01_verify','interface_suite02','interface_suite03','runtime_green','experiment01_register','experiment01_driver'):
        assert (E/(n+'.exit')).read_text().strip()=='0',n
    assert (E/'interface_suite.exit').read_text().strip()!='0'
    means={};uuids={};pids=[];decisions=0;train_decisions=0;terminals=0;maximum_error=0.
    for block,arm in itertools.product(range(8),ARMS):
        out=EXP/f'b{block:02d}_{arm}';r=read(out/'report.json');v=read(out/'independent_verification.json');identity=read(out/'identity.json')
        assert r['passed'] and v['passed'] and not r['pilot'] and not r['stimulation'] and not r['source_full_qc_passed'] and not r['biological_validated']
        assert r['block']==v['block']==block and r['arm']==v['arm']==arm and r['collection_pid']==v['collection_pid']!=v['verification_pid']
        assert r['registration_sha256']==v['registration_sha256']==reg_sha and v['collection_manifest_sha256']==sha(out/'collection_manifest.json')
        assert identity['initial']==r['update']['before']==reg['common_initial'] and identity['source']==reg['source_identity'] and identity['config']==reg['configs'][f'{block}/{arm}']
        assert r['gpu']['physical_gpu']==block and r['gpu']['uuid']==v['gpu']['uuid'];uuids.setdefault(block,r['gpu']['uuid']);assert uuids[block]==r['gpu']['uuid']
        pids.extend([r['collection_pid'],v['verification_pid']]);assert all(x is True for x in r['update']['checks'].values())
        assert 1<=r['update']['metrics']['optimizer_steps']<=8 and r['update']['critic_steps']==64 and v['actual_weight_optimizer_checks']
        errors=[r['update']['reload_error'],v['updated_train_statistic_error']]+[q[k] for q in v['phases'] for k in ('statistic_error','bootstrap_error')]
        assert all(math.isfinite(x) and 0<=x<=1e-6 for x in errors);maximum_error=max(maximum_error,*errors)
        phase_means={'pre':[],'post':[]}
        for j,name in enumerate(('pre0','pre1','train','post0','post1')):
            phase=out/name;events=[json.loads(s) for s in (phase/'rollout.jsonl').read_text().splitlines()];rows=[e['training'] for e in events];plan=reg['schedules'][f'{block}/{arm}'][j];boot=read(phase/'bootstrap.json')
            assert read(phase/'assignment.json')==plan and v['phases'][j]['phase']==name and v['phases'][j]['decisions']==len(rows)
            assert all(v['phases'][j][k] is True for k in ('native_replay','snapshot_inputs_verified','reward_gae_verified'))
            assert len(rows)==128 if name=='train' else 1<=len(rows)<=96
            audit_gae(rows,boot['value']);tick=0;value=0.;raw=0.;expected_seed=plan['world_seed'];expected_obs=None
            for i,e in enumerate(events):
                row=e['training'];r1=e['r1_score']['reward'];u=e['snapshot']['score']['u'];weight=.05 if name=='train' and arm in ('reward','both','shuffled') else 0.
                assert e['index']==i and e['window']==plan['order'][i] and e['seed']==expected_seed
                if expected_obs is not None:assert e['observation']==expected_obs
                assert row['reward']==r1+weight*u and e['reward_record']['aux_weight']==weight and e['reward_record']['r1_weight']==1.
                assert e['prepared']['branches'][row['verb']][row['target']]==e['action'] and row['duration']==e['r1_score']['elapsed_ticks']
                assert row['terminal']==e['terminal']
                if row['terminal']:
                    terminals+=1
                    if name=='train':expected_seed+=1;expected_obs=e['reset_observation'];assert e['reset_seed']==expected_seed
                    else:assert i==len(rows)-1 and e['reset_seed'] is None
                else:expected_obs=e['next_observation'];assert e['reset_seed'] is None
                value+=.99985**tick*r1;tick+=row['duration'];raw+=r1
            m=read(phase/'metrics.json');assert abs(value-m['discounted_r1'])<1e-12 and raw==m['r1_sum'] and tick==m['ticks']
            assert m['final_observation']==events[-1]['next_observation'] and m== (r['train'] if name=='train' else r[name[:-1]][int(name[-1])])
            if name=='train':train_decisions+=len(rows)
            else:phase_means[name[:-1]].append(value)
            decisions+=len(rows)
        means[block,arm]={p:statistics.mean(xs) for p,xs in phase_means.items()}
    assert len(set(uuids.values()))==8 and train_decisions==5120 and decisions<=20480
    analysis=read(EXP/'analysis.json');assert analysis['status']=='complete'
    for a,b,primary in [('both','baseline',True),('both','shuffled',True),('state','baseline',False),('reward','baseline',False),('both','state',False)]:
        d=[means[i,a]['post']-means[i,b]['post'] for i in range(8)];g=[d[i]-(means[i,a]['pre']-means[i,b]['pre']) for i in range(8)]
        for phase,values in [('post',d),('gain',g)]:
            x=statistics_for(values);y=analysis['primary' if primary else 'secondary'][a+'-'+b][phase]
            assert max(abs(q-w) for q,w in zip(values,y['differences']))<1e-12 and abs(x['mean']-y['mean_difference'])<1e-12 and x['p']==y['p_two_sided']
            assert max(abs(q-w) for q,w in zip(x['ci'],y['bootstrap_95']))<1e-12
    telemetry=[json.loads(s) for s in (EXP/'gpu_telemetry.jsonl').read_text().splitlines()]
    for pid in pids:
        assert any(any(int(line.split(',')[0])==pid for line in t.get('compute','').splitlines()) for t in telemetry),('missing GPU PID telemetry',pid)
    assert read(EXP/'independent_analysis.json')['collection_decisions']==decisions
    result=dict(passed=True,files=len(manifest)+1,cells=40,physical_gpus=8,train_decisions=train_decisions,collection_decisions=decisions,natural_terminals=terminals,
        maximum_statistic_error=maximum_error,biological_recordings=1,failed_attempts_preserved=True,telemetry_samples=len(telemetry),telemetry_error_samples=sum('error' in t for t in telemetry),registration_sha256=reg_sha)
    with (ROOT/'download_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
