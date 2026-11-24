import copy
import importlib
from pathlib import Path
import tempfile
import unittest

from agent import m9_contract
from agent.mea_contract import observation_key
from agent.mea_replay import AuditedRecording
from tests.test_mea_p0 import observation, GODOT, ROOT
from tests.test_mea_replay import fixture


class AblationTests(unittest.TestCase):
    def test_fixed_split_and_derangement(self):
        m=importlib.import_module('agent.mea_ablation')
        p=m.protocol()
        self.assertEqual(p['train_indices'],list(range(33)))
        self.assertEqual(p['eval_indices'],list(range(36,57)))
        self.assertEqual(p['blocks'],8)
        for b in range(8):
            for split,indices in [('train',p['train_indices']),('eval',p['eval_indices'])]:
                a=m.assignment('shuffled',split,b)
                self.assertEqual(sorted(a),indices)
                self.assertTrue(all(x!=y for x,y in zip(a[:-1],indices[:-1])))
                self.assertEqual(a[-1],indices[-1])
                self.assertEqual(sorted(a[:-1]),indices[:-1])
                self.assertEqual(a,m.assignment('shuffled',split,b))
                self.assertEqual(m.assignment('real',split,b),indices)
        with self.assertRaises(ValueError):m.assignment('live','train',0)
        with self.assertRaises(ValueError):m.assignment('real','train',8)

    def test_none_is_blind_and_real_shuffled_share_text(self):
        m=importlib.import_module('agent.mea_ablation')
        obs=observation()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);h=fixture(root);data=AuditedRecording(root,manifest_sha256=h)
            s=data.snapshot(0,observation_key(obs))
            none=m.encode_arm(obs,None,'none')
            self.assertEqual(none['messages'],m9_contract.encode(obs,'reserves')['messages'])
            self.assertEqual(none['critic_input'][-4:],[0.]*4)
            self.assertIsNone(none['wetware_snapshot'])
            self.assertEqual(len(none['critic_input']),112)
            with self.assertRaises(ValueError):m.encode_arm(obs,s,'none')
            a=m.encode_arm(obs,s,'real');b=m.encode_arm(obs,s,'shuffled')
            self.assertEqual(a['messages'],b['messages'])
            self.assertEqual(a['critic_input'],b['critic_input'])
            self.assertEqual(a['critic_input'][-4:],list(s.feature_values))

    def test_paired_statistics_are_block_level_and_complete(self):
        m=importlib.import_module('agent.mea_ablation')
        cells=[dict(block=b,arm=a,pre=[{'discounted_r1':0.}]*2,
                    post=[{'discounted_r1':float(a=='real')}]*2)
               for b in range(8) for a in m.ARMS]
        r=m.analyze(cells)
        self.assertEqual(r['replication_units'],8)
        for c in r['contrasts'].values():
            self.assertEqual(c['post']['mean_difference'],1.)
            self.assertEqual(c['post']['bootstrap_95'],[1.,1.])
            self.assertEqual(c['post']['p_two_sided'],2/256)
            self.assertEqual(c['post']['p_bonferroni_two_contrasts'],4/256)
        self.assertEqual(m.analyze(cells[:-1])['status'],'incomplete')
        with self.assertRaises(ValueError):m.analyze(cells+[cells[0]])
        for c in cells:c['post']=[{'discounted_r1':0.}]*2
        r=m.analyze(cells)
        self.assertEqual(r['contrasts']['real_minus_none']['post']['p_two_sided'],1.)
        cells[0]['post'][0]['discounted_r1']=float('nan')
        with self.assertRaises(ValueError):m.analyze(cells)

    def test_discounted_return_does_not_use_bootstrap(self):
        m=importlib.import_module('agent.mea_ablation')
        after=dict(dead=False,health=90,hunger=70,thirst=65,inventory={'berries':1},visited_count=3)
        rows=[dict(reward=1.,duration=2,terminal=False,transition={'after':after}),
              dict(reward=2.,duration=1,terminal=False,transition={'after':after})]
        r=m.episode_metrics(rows,gamma=.5)
        self.assertEqual(r['discounted_r1'],1.5)
        self.assertEqual(r['undiscounted_r1'],3.)
        self.assertEqual(r['elapsed_ticks'],3)

    def test_copied_and_stale_verification_receipts_rejected(self):
        m=importlib.import_module('tools.mea_ablation')
        good=dict(passed=True,model_verified=True,block=0,arm='real',registration_sha256='r',manifest_sha256='m',
            native_episodes=5,statistic_errors={k:0. for k in ['pre0','pre1','train','post0','post1']})
        m.validate_receipt(good,0,'real','r','m')
        for key,value in [('block',1),('arm','none'),('registration_sha256','other'),('manifest_sha256','stale'),
                          ('statistic_errors',{}),('model_verified',False)]:
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_receipt(good|{key:value},0,'real','r','m')

    def test_common_start_and_actual_update_checks(self):
        m=importlib.import_module('tools.mea_ablation')
        baseline={k:k for k in ['actor','critic','base','reference']}
        m.validate_common_start(baseline,baseline)
        for k in baseline:
            with self.assertRaises(ValueError):m.validate_common_start(baseline|{k:'different'},baseline)
        checks={k:True for k in ['actor_changed','critic_changed','base_unchanged','reference_unchanged',
            'inputs_unchanged','base_no_gradient','neural_columns_as_expected']}
        update=dict(checks=checks,metrics={'optimizer_steps':2},saved_statistic_error=0.)
        m.validate_update_claim(update)
        for changes in [dict(checks={}),dict(checks=checks|{'actor_changed':False}),
                        dict(saved_statistic_error=float('nan')),dict(metrics={'optimizer_steps':1})]:
            with self.assertRaises(ValueError):m.validate_update_claim(update|changes)
        loaded=dict(actor='new_actor',critic='new_critic',base='base',reference='reference')
        m.validate_loaded_change(baseline,loaded,False,'none',{'actor':[2],'critic':[16]})
        for state,columns,steps in [(baseline,False,{'actor':[2],'critic':[16]}),
            (loaded,True,{'actor':[2],'critic':[16]}),(loaded,False,{'actor':[1],'critic':[16]})]:
            with self.assertRaises(ValueError):m.validate_loaded_change(baseline,state,columns,'none',steps)

    @unittest.skipUnless(GODOT.exists(),'native Godot required')
    def test_native_roundtrip_and_none_metadata_tampering(self):
        m=importlib.import_module('tools.mea_ablation')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);s=root/'source';s.mkdir();h=fixture(s)
            data=AuditedRecording(s,manifest_sha256=h)
            for arm in ['none','real','shuffled']:
                order=[0,2,1] if arm=='shuffled' else [0,1,2]
                rows,boot,metrics=m.episode(None,data,arm,order,123,456,root/arm,GODOT,ROOT)
                self.assertTrue(m.replay_episode(rows,boot,metrics,data,arm,order,123,GODOT,ROOT))
                changed=copy.deepcopy(rows);changed[0]['prepared']['critic_input'][-1]+=1
                with self.assertRaises(ValueError):m.replay_episode(changed,boot,metrics,data,arm,order,123,GODOT,ROOT)
                self.assertTrue(all(r['reward']==r['r1_score']['reward'] for r in rows))


try:
    import torch
    HAS_TORCH=True
except ImportError:HAS_TORCH=False


@unittest.skipUnless(HAS_TORCH,'GPU ML environment required')
class AblationPolicyTests(unittest.TestCase):
    def test_prepare_none_never_receives_features_and_keeps_tokens(self):
        from types import SimpleNamespace
        from agent.mea_ablation_policy import prepare_arm
        obs=observation()
        fake=SimpleNamespace(arm='none',c=SimpleNamespace(map_mode=obs['map_mode'],navigation_steps=obs['navigation_steps']),
            _tokens=lambda messages:[11,22,33],tokenizer=SimpleNamespace(encode=lambda s,**kw:list(map(ord,s)),eos_token_id=5))
        rows=prepare_arm(fake,[obs],[None])
        self.assertEqual(rows[0]['input_ids'],[11,22,33])
        self.assertEqual(rows[0]['critic_input'][-4:],[0.]*4)
        with self.assertRaises(ValueError):prepare_arm(fake,[obs],[])
        with self.assertRaises(ValueError):prepare_arm(fake,[obs],[object()])


if __name__=='__main__':unittest.main()
