"""Verify shipped receipts and reproduce the endpoints named by the written draft."""
import csv
import hashlib
import json
from math import ceil, isclose
from pathlib import Path
import tarfile
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE.parent
STUDY=RESULT.parent
DRAFT=STUDY/'offline-confirmatory-registration'
PRIOR=STUDY/'offline-stress-three-arms'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(raw):return hashlib.sha256(raw).hexdigest()


def parse_manifest(path):
    pairs=[line.split('  ',1) for line in path.read_text('utf-8').splitlines()]
    out={name:pin for pin,name in pairs}
    assert len(out)==len(pairs)
    return out


def csvrows(file):
    with file.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def band(mat,rng_pair,alpha):
    seed_weights,world_weights=rng_pair
    left=seed_weights @ mat
    draws=(left*world_weights).sum(axis=1)/mat.size
    lo,hi=np.quantile(draws,[alpha/2,1-alpha/2]);by_seed=mat.mean(axis=1)
    win=int((by_seed>0).sum());lose=int((by_seed<0).sum());thresh=ceil(.75*len(by_seed))
    return dict(delta=float(mat.mean()),lo=float(lo),hi=float(hi),favorable=win,adverse=lose,stable_gain=bool(lo>0 and win>=thresh),stable_harm=bool(hi<0 and lose>=thresh))


def weights(ns,nw,draws,seed):
    rng=np.random.default_rng(seed)
    return rng.multinomial(ns,np.full(ns,1/ns),size=draws),rng.multinomial(nw,np.full(nw,1/nw),size=draws)


def effects(spec,reg,joined,comparison,subset,assert_reported):
    worlds=spec['worlds'];seeds=subset
    w=weights(len(seeds),len(worlds),reg['analysis']['bootstrap_resamples'],spec['bootstrap_seed'])
    summaries=[]
    for family in ['primary_endpoints','secondary_endpoints']:
        alpha=reg['analysis']['primary_two_sided_alpha_per_comparison'] if family=='primary_endpoints' else reg['analysis']['secondary_alpha']
        for endpoint in reg[family]:
            scene=endpoint['scene'];metric=endpoint['metric'];direction=-1 if metric.endswith('_wait') else 1
            for control in ['masked','shuffled']:
                m=np.array([[direction*(int(joined[scene,s,g]['ordered_'+metric])-int(joined[scene,s,g][control+'_'+metric])) for g in worlds] for s in seeds],dtype=np.int64)
                effect=band(m,w,alpha)
                if assert_reported:
                    old=comparison['effects'][scene][control][metric]
                    expected=(old['delta'],old['lo'],old['hi']) if direction==1 else (-old['delta'],-old['hi'],-old['lo'])
                    assert all(isclose(effect[k],val,abs_tol=1e-12,rel_tol=0) for k,val in zip(['delta','lo','hi'],expected))
                    assert effect['stable_gain']==old['stable_gain'] and effect['stable_harm']==old['stable_harm'] and effect['favorable']==old['favorable_seeds']
                summaries.append(dict(family=family,scene=scene,metric=metric,control=control,interval_level=1-alpha,**effect))
    return summaries


def main():
    allpins=parse_manifest(HERE/'SHA256SUMS.evaluated');flat=parse_manifest(HERE/'SHA256SUMS.flat')
    for f,h in flat.items():assert sha((RESULT/f).read_bytes())==h
    for f,h in parse_manifest(HERE/'transfer.sha256').items():
        with (HERE/f).open('rb') as src:assert hashlib.file_digest(src,'sha256').hexdigest()==h
    reg=json.loads((DRAFT/'design.json').read_bytes());spec=json.loads((RESULT/'design.json').read_bytes());run_pin=digest(spec)
    archive=HERE/'evaluated-evidence.tar.gz';seen=set(); archive_h=allpins.copy();archive_h['verification-20261120/SHA256SUMS.evaluated']=sha((HERE/'SHA256SUMS.evaluated').read_bytes())
    eps=csvrows(RESULT/'episodes.csv');pair_rows=csvrows(RESULT/'paired.csv')
    episodes={(r['scene'],int(r['seed']),int(r['world']),r['arm']):r for r in eps}
    paired={(r['scene'],int(r['seed']),int(r['world'])):r for r in pair_rows}
    seeds=spec['training_seeds'];worlds=spec['worlds'];arms=spec['arms'];scenes=list(spec['scenes'])
    assert len(set(seeds))==len(seeds)==32 and len(set(worlds))==len(worlds)==64 and set(arms)=={'masked','ordered','shuffled'}
    triples={(scene,s,w) for s in seeds for w in worlds for scene in scenes}
    assert len(paired)==len(pair_rows)==len(triples)==6144 and paired.keys()==triples
    expected={(*key,a) for key in triples for a in arms}; assert episodes.keys()==expected and len(episodes)==len(eps)==18432
    pilot_rows=csvrows(PRIOR/'stress'/'paired.csv');pseeds={int(r['seed']) for r in pilot_rows};pmaps={int(r['world']) for r in pilot_rows}
    assert sha((PRIOR/'stress/paired.csv').read_bytes())==reg['source_paired_data_sha256']
    comparison=json.loads((RESULT/'comparison.json').read_bytes());training_audits=json.loads((RESULT/'training-audit.json').read_bytes()); fits={(r['seed'],r['arm']):r for r in training_audits}
    assert len(fits)==len(training_audits)==96 and len({digest(r['initial']) for r in fits.values()})==len({r['config_digest'] for r in fits.values()})==1
    ctx=json.loads((PRIOR/'context.json').read_bytes())
    assert allpins['context.json']==sha((PRIOR/'context.json').read_bytes())
    train_spec=None; model_records={};game_keys=set();freeze_keys=set();train_groups=set();decisions=r2_positives=0
    registration_pin=sha((DRAFT/'design.json').read_bytes()); assert registration_pin==allpins['registration-design.json']
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=member.name
            assert member.isfile() and name not in seen and name in archive_h
            seen.add(name);raw=tar.extractfile(member).read();assert sha(raw)==archive_h[name],name
            parts=name.split('/')
            if name=='design.json':train_spec=json.loads(raw)
            elif parts[:2]==['stress','games']:
                _,_,seed_s,scene,arm,file=parts;seed=int(seed_s);d=json.loads(raw); frozen=fits[seed,arm]['after']
                assert d['spec']==run_pin and d['seed']==seed and d['scene']==scene and spec['states'][f'{seed}/{arm}']==frozen
                if file=='frozen.json':
                    key=scene,seed,arm; assert key not in freeze_keys and d['worlds']==64 and d['arm']==arm and d['state']==frozen;freeze_keys.add(key);continue
                world=int(file[:-5]);key=scene,seed,world,arm;assert key not in game_keys;game_keys.add(key)
                g=d['game'];r=episodes[key];pr=paired[scene,seed,world]; rr=g['rows'];n=len(rr)
                assert g['world']==world and g['mode']==arm and d['frozen']==frozen and int(r['decisions'])==n==g['decisions']
                assert all(row['r2_contribution']==0 for row in rr);r2_positives+=sum(row['r2_logged'] for row in rr);decisions+=n
                assert int(r['r2_logged'])==sum(row['r2_logged'] for row in rr) and float(r['r1'])==g['r1']
                end=g['end_tick']; alive=g['survived']; assert end<=spec['horizon'] and (end==spec['horizon'] or not alive)
                for m,val in [('survived',alive),('end_tick',end),('cells',g['new_cells']),('drink_wait',g['restricted_wait']['drink']),('eat_wait',g['restricted_wait']['eat']),('drank',int(g['first_success_tick']['drink'] is not None)),('ate',int(g['first_success_tick']['eat'] is not None))]:assert int(pr[arm+'_'+m])==int(r[m])==val
                for verb in ['drink','eat']:
                    tick=g['first_success_tick'][verb];value='' if tick is None else str(tick)
                    assert r[verb+'_tick']==pr[arm+'_'+verb+'_tick']==value and g['restricted_wait'][verb]==(60 if tick is None else tick)
            elif parts[0]=='models':
                seed=int(parts[1]);arm=parts[2];kind=parts[-1];key=seed,arm
                if kind.startswith('train-'):
                    batch=int(kind[6:-6]);g=(seed,arm,batch);assert g not in train_groups; train_groups.add(g)
                    rows=[json.loads(line) for line in raw.splitlines()]; assert len(rows)==12
                    order=train_spec['training_order'][str(seed)] if arm=='shuffled' else list(range(48))
                    for i,r in enumerate(rows):
                        window=ctx['series']['train'][order[(batch-1)*12+i]];sn=r['snapshot'];pr=r['prepared'];vec=[0.]*4 if arm=='masked' else window['normalized']
                        assert r['seed']==train_spec['episodes'][str(seed)][batch-1] and r['index']==i and sn['window_index']==window['index'] and sn['source_pin']==ctx['source_sha256'] and sn['events_sha256']==window['events_sha']
                        assert pr['wetware_features']==pr['critic_input'][-4:]==vec and ','.join(f'{k}={v:.3f}' for k,v in zip(sn['names'],vec)) in pr['messages'][1]['content']
                        assert r['contribution']==0 and r['r1_score']['reward']==r['reward']
                else:model_records.setdefault(key,{})[kind]=json.loads(raw)
    assert seen==archive_h.keys() and game_keys==expected and freeze_keys=={(c,s,a) for c in scenes for s in seeds for a in arms}
    assert len(train_groups)==384 and model_records.keys()==fits.keys() and decisions==comparison['decisions']==669715
    assert spec['training_audit_sha']==flat['training-audit.json'] and digest(train_spec)==spec['training_design_sha'] and spec['beta']==train_spec['beta']==0 and spec['alpha']==train_spec['alpha']==1 and digest(ctx)==train_spec['context_sha']
    assert sha((PRIOR/'stress/analyze.py').read_bytes())==train_spec['analysis_code_sha']
    assert allpins['stress/analyze.py']==sha((PRIOR/'stress/analyze.py').read_bytes().replace(b'len(seeds)) and (',b'sw.shape[1]) and ('))
    for f,h in [('train.py','training_code_sha'),('play60.py','eval_code_sha'),('stress/evaluate.py','stress_eval_code_sha')]:assert allpins[f]==train_spec[h]
    for key,records in model_records.items():
        seed,arm=key;man=records['manifest.json'];training=records['registration.json'];update=records['update.json'];aud=fits[key]
        assert training['training_seed']==seed and training['arm']==arm and training['study_digest']==digest(train_spec) and digest(training)==man['registration_sha256'] and training['windows_digest']==digest(ctx['series'])
        assert training['initial']==update['initial']==aud['initial'] and update['after']==aud['after']==spec['states'][f'{seed}/{arm}'] and man['actor_sha256']==aud['after']['actor']
        assert aud['checkpoint_manifest']==allpins[f'models/{seed}/{arm}/checkpoint/manifest.json'] and aud['reward_contribution']==0 and aud['steps']==48
        assert update['actor_steps']==[aud['actor_steps']]==[8] and update['critic_steps']==[aud['critic_steps']]==[64] and len(update['rounds'])==4 and all(r['decisions']==12 and r['contribution']==0 for r in update['rounds'])
    info=json.loads((RESULT/'design-repair.json').read_bytes());raw=(RESULT/'design.json').read_bytes();reconstructed=raw.replace(info['pilot_sha_after'].encode(),info['pilot_sha_before'].encode());assert sha(reconstructed)==info['design_sha_before'] and digest(json.loads(reconstructed))==info['spec_before'] and sha(raw)==info['design_sha_after'] and digest(spec)==info['spec_after'] and info['result_files_updated']==len(game_keys)+len(freeze_keys)
    assert set(seeds)&pseeds==pseeds and set(worlds)==pmaps
    independent_ids=[s for s in seeds if s not in pseeds];train_maps={x for vals in train_spec['episodes'].values() for x in vals};assert not train_maps.intersection(worlds)
    primary=effects(spec,reg,paired,comparison,seeds,True); sensitivity=effects(spec,reg,paired,comparison,independent_ids,False)
    report=dict(archive_members_checked=len(seen),original_evidence_files_checked=len(allpins),flat_files_verified=len(flat),training_models=96,training_steps=4608,training_maps=len(train_maps),training_evaluation_map_overlap=len(train_maps.intersection(worlds)),masked_actor_critic_input_verified=True,cells_measure='newly revealed game tiles',episode_rows=len(game_keys),paired_rows=len(paired),frozen_receipts=len(freeze_keys),decisions=decisions,evaluation_diagnostic_events=r2_positives,evaluation_r2_contribution=0,bootstrap_resamples=spec['draws'],original_registration_file_sha256=registration_pin,training_registration_digest_field=train_spec['registration_sha256'],original_design_hash_reconstructed=True,prior_result_byte_identity_before_edit='unverified',analysis_change='two references to out-of-scope len(seeds) replaced with sw.shape[1]',threshold_training_seeds=24,pilot_training_seeds_reused=len(pseeds),pilot_worlds_reused=len(pmaps),new_training_seeds=independent_ids,all32=primary,additional24_posthoc=sensitivity)
    (HERE/'local-verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Verified',len(allpins),'archived evidence files and',len(primary),'registered comparisons. Primary gain count',sum(r['stable_gain'] for r in primary if r['family']=='primary_endpoints'))
    print('Additional 24 training seeds, primary',[(r['scene'],r['control'],round(r['delta'],5),round(r['lo'],5),round(r['hi'],5),r['favorable']) for r in sensitivity if r['family']=='primary_endpoints'])

if __name__=='__main__':main()
