import copy
import tempfile
from pathlib import Path
import unittest
import json
from unittest.mock import patch
from tools.compare_m8x_survival import check_audit,check_ledger,build_result
from agent.m8x_identity import append_ledger_event
from agent.m8x_identity import config_sha,BASE_WEIGHT_SHA
from agent.m8_policy_identity import PARENT_ACTOR_SHA256
from agent.m8x_eval import episode_seed,DECODING
from tools.rl_audit_io import file_sha256
from tools.m8x_metrics import qualify_selection


def full_fixture(root,attempt=1,candidate_ticks=1800):
    """Synthetic evidence to test schema plumbing, never a real audit claim."""
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8');return str(path)
    root.mkdir(parents=True,exist_ok=True)
    (root/'frozen_source.zip').write_bytes(b'synthetic archive only')
    cfg={'output':str(root.resolve()),'project':str(root.resolve()),'attempt':attempt,'rng_prefix':'m8_exploration_reliable_v1','validation_seeds':list(range(16)),'test_seeds':list(range(100,200))}
    reg=dict(archive_sha256=file_sha256(root/'frozen_source.zip'),balance_sha256='b'*64,config=cfg,source_hashes={'synthetic':'s'},source_sha256={'synthetic':'s'},training_reset='natural death only',
        execution_identity={'config_sha256':config_sha(cfg),'base_weight_sha256':BASE_WEIGHT_SHA,'runtime':{'fixture':True,'kernel_identity':{'fixture':True}},'tokenizer_files_sha256':{'fixture':'a'},'parent_restore_sha256':{'fixture':'b'}})
    write('registration.json',reg);write('preregistration.json',reg);regsha=file_sha256(root/'registration.json')
    write('transitions.jsonl',{'synthetic':True});trace=file_sha256(root/'transitions.jsonl')
    final_actor='3'*64
    summary=dict(decisions=98304,world_ticks=40000,initial_actor_sha256=PARENT_ACTOR_SHA256,final_actor_sha256=final_actor,
        source_counts={'model':98304,'fallback':0,'scripted':0},weight_change_verified=True)
    write('training_summary.json',summary)
    (root/'training.jsonl').write_text('\n'.join(json.dumps(dict(update=i,decisions=i*2048,ticks=40000 if i==48 else i*1000)) for i in range(1,49)),encoding='utf-8')
    training=write('training_audit.json',dict(passed=True,complete_training_budget=True,prefix_only=False,replayed_decisions=98304,scored_world_ticks=40000,
        trace_sha256=trace,source_sha256=reg['source_hashes'],registration_sha256=regsha))
    finished=write('finished.json',{'exit_code':0})
    selection={};reports={};ledger_evidence={}
    for index,name in enumerate(['original_v2','balanced_parent',*[f'checkpoint_{u:04d}' for u in (16,32,48)]]):
        arm=name if index<2 else 'candidate';actor=PARENT_ACTOR_SHA256 if index<2 else str(index-1)*64
        ticks=1000 if index<2 else candidate_ticks
        episodes=[dict(seed=s,replicate=r,survival_ticks=ticks,terminal=ticks<1800,alive_at_horizon=ticks==1800,decisions=1,
            rng_seed=episode_seed(cfg['rng_prefix'],'open1' if arm=='original_v2' else 'open3',s,r),source_counts={'model':1,'fallback':0},actions={'wait':1}) for s in range(16) for r in (0,)]
        tracepath=root/(name+'.jsonl');tracepath.write_text('synthetic only\n',encoding='utf-8')
        report=dict(arm=arm,actor_sha256=actor,contract='open_world_hierarchical_v2' if arm=='original_v2' else 'open_world_exploration_v1',
            profile='open1' if arm=='original_v2' else 'open3',world_version=2 if arm=='original_v2' else 5,environment_version=None if arm=='original_v2' else 'm8_exploration_v1',balance_sha256=None if arm=='original_v2' else 'b'*64,horizon=1800,temperature=1,decoding=DECODING,source_sha256=reg['source_hashes'],rng_prefix=cfg['rng_prefix'],
            seeds=cfg['validation_seeds'],replicates=[0],episodes=episodes,mean_survival_ticks=ticks,alive_at_horizon=16 if ticks==1800 else 0,trace_sha256=file_sha256(tracepath))
        rp=write(name+'.json',report);rsha=file_sha256(Path(rp));ledger_evidence[Path(rp).name]=rsha;reports[name]=report
        audit=dict(passed=True,trace_sha256=report['trace_sha256'],actor_sha256=actor,episodes=16,replayed_decisions=16,
            report_sha256=rsha,registration_sha256=regsha,source_sha256=reg['source_hashes'],prefix_only=False,rng_sampling_reproduced=True,runtime={'fixture':True,'kernel_identity':{'fixture':True}})
        selection[name]=dict(report=rp,replay_audit=write(name+'_replay.json',audit),rng_audit=write(name+'_rng.json',audit))
        if index>=2:write(name+'/policy.json',dict(world_version=5,environment_version='m8_exploration_v1',balance_sha256='b'*64,contract='open_world_exploration_v1',reward_config=__import__('agent.m8x_contract',fromlist=['REWARD_CONFIG']).REWARD_CONFIG,kernel_identity={'fixture':True},reference_actor_sha256=PARENT_ACTOR_SHA256,actor_sha256=actor))
    chosen=qualify_selection({u*2048:reports[f'checkpoint_{u:04d}']['episodes'] for u in (16,32,48)},reports['balanced_parent']['episodes'])
    write('selection.json',dict(chosen,selected_checkpoint=str(root/'checkpoint_0016'),selected_actor_sha256='1'*64))
    se=dict(attempt=attempt,phase='selection',status='started',registration_sha256=regsha,selected_checkpoint=None,selected_actor_sha256=None,evidence_sha256={})
    append_ledger_event(root/'ledger_selection.jsonl',se);append_ledger_event(root/'ledger_selection.jsonl',dict(se,status='completed',evidence_sha256=dict(ledger_evidence)))
    ledger_evidence.update({name:file_sha256(root/name) for name in ('transitions.jsonl','training_summary.json','selection.json')})
    ledger=root/'ledger_train.jsonl';event=dict(attempt=attempt,phase='train',status='started',registration_sha256=regsha,selected_checkpoint=None,selected_actor_sha256=None,evidence_sha256={})
    append_ledger_event(ledger,event);append_ledger_event(ledger,dict(event,status='completed',evidence_sha256=ledger_evidence))
    return dict(phase='selection',training_run=str(root),training_audit=training,training_finished=finished,training_ledger=str(ledger),selection=selection)


class BalancedGateTests(unittest.TestCase):
    def test_complete_synthetic_selection_json_roundtrip_and_missing_audit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"output/runtime_tests") as tmp,patch('tools.compare_m8x_survival.verify_registration'):
            root=Path(tmp);request=full_fixture(root)
            r=build_result(request)
            self.assertTrue(r['eligible_for_test']);self.assertFalse(r['release_authorized'])
            self.assertEqual(r,json.loads(json.dumps(r)))
            p=Path(request['selection']['balanced_parent']['rng_audit']);v=json.loads(p.read_text());v['rng_sampling_reproduced']=False;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):build_result(request)
    def test_failed_exit_and_old_world_report_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"output/runtime_tests") as tmp,patch('tools.compare_m8x_survival.verify_registration'):
            request=full_fixture(Path(tmp));p=Path(request['training_finished']);p.write_text('{"exit_code":1}')
            with self.assertRaises(ValueError):build_result(request)

    def test_full_synthetic_final_release_requires_all_three_audited_arms(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"output/runtime_tests") as tmp,patch('tools.compare_m8x_survival.verify_registration'):
            root=Path(tmp);request=full_fixture(root);request['phase']='test';request['test']={};evidence={}
            regsha=file_sha256(root/'registration.json')
            for arm in ('original_v2','balanced_parent','candidate'):
                base='checkpoint_0016' if arm=='candidate' else arm
                report=json.loads(Path(request['selection'][base]['report']).read_text())
                prototype=report['episodes'][0];report['seeds']=list(range(100,200));report['replicates']=[0]
                report['episodes']=[dict(prototype,seed=s,replicate=0,rng_seed=episode_seed('m8_exploration_reliable_v1',report['profile'],s,0)) for s in report['seeds']]
                report['alive_at_horizon']=100 if arm=='candidate' else 0
                rp=root/('test_'+arm+'.json');tp=rp.with_suffix('.jsonl');tp.write_text('synthetic only\n');report['trace_sha256']=file_sha256(tp)
                rp.write_text(json.dumps(report));rsha=file_sha256(rp);evidence[rp.name]=rsha
                audit=json.loads(Path(request['selection'][base]['rng_audit']).read_text())
                audit.update(episodes=100,replayed_decisions=100,report_sha256=rsha,trace_sha256=report['trace_sha256'])
                ap=root/('test_'+arm+'_audit.json');ap.write_text(json.dumps(audit))
                request['test'][arm]={'report':str(rp),'replay_audit':str(ap),'rng_audit':str(ap)}
            ledger=root/'ledger_test.jsonl';event=dict(attempt=1,phase='test',status='started',registration_sha256=regsha,selected_checkpoint='checkpoint_0016',selected_actor_sha256='1'*64,evidence_sha256={})
            append_ledger_event(ledger,event);append_ledger_event(ledger,dict(event,status='completed',evidence_sha256=evidence))
            request.update(test_ledger=str(ledger),test_finished=request['training_finished'])
            result=build_result(request);self.assertTrue(result['release_authorized']);self.assertEqual(result['decision'],'accept')
            self.assertEqual(json.loads(json.dumps(result)),build_result(request))
            altered=copy.deepcopy(request);del altered['test']['original_v2']
            with self.assertRaisesRegex(ValueError,'Three'):build_result(altered)


    def test_authorize_rebuilds_persisted_selection_and_rejects_changes(self):
        from agent.m8x_eval import authorize_test
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"output/runtime_tests") as tmp,patch('tools.compare_m8x_survival.verify_registration'),patch('agent.m8x_registration.verify_registration'):
            root=Path(tmp);run=root/'run';request=full_fixture(run);review=build_result(request)
            req=root/'review_request.json';rev=root/'review.json';req.write_text(json.dumps(request));rev.write_text(json.dumps(review))
            authorization=dict(run=str(run),mode='test',output=str(root/'test'),selection_review=str(rev),selection_review_request=str(req))
            result=authorize_test(authorization);self.assertEqual(result[2],review)
            for field,value in [('selected_actor_sha256','f'*64),('selected_checkpoint','../checkpoint_0016'),('eligible_for_test',False)]:
                changed=dict(review,**{field:value});rev.write_text(json.dumps(changed))
                with self.subTest(field=field),self.assertRaises(ValueError):authorize_test(authorization)
            rev.write_text(json.dumps(review))
            with self.assertRaises(ValueError):authorize_test(dict(authorization,output=str(run/'test')))
            with self.assertRaises(ValueError):authorize_test(dict(authorization,mode='validation'))
            altered=dict(request,training_run=str(root/'wrong'));req.write_text(json.dumps(altered))
            with self.assertRaises((ValueError,FileNotFoundError)):authorize_test(authorization)
            req.write_text(json.dumps(request))
            append_ledger_event(run/'ledger_test.jsonl',dict(attempt=1,phase='test',status='started',registration_sha256=review['registration_sha256'],selected_checkpoint='checkpoint_0016',selected_actor_sha256=review['selected_actor_sha256'],evidence_sha256={}))
            with self.assertRaises(FileExistsError):authorize_test(authorization)

if __name__=='__main__':unittest.main()
