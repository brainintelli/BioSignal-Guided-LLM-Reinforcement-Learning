"""Behavioral checks for the registered supply-margin experiment, not model scores."""
import copy
import importlib.util
import unittest

from tests.test_m8s3_contract import observation
from agent.m8s3_contract import encode, SurvivalReward as OriginalReward


def snapshot(obs,dead=False):
    p=obs['player']
    return dict(tick=obs['tick'],dead=dead,health=p['health'],hunger=p['hunger'],thirst=p['thirst'],
                player={'x':p['x'],'y':p['y']})


def step(before,after,dead=False):
    return dict(before=snapshot(before),after=snapshot(after,dead),elapsed_ticks=1,trace=[snapshot(after,dead)])


class SupplyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8s3_supply'),'Supply reward and critic facts must exist')
        from agent import m8s3_supply
        return m8s3_supply

    def food_state(self,dx=5,hunger=20,food=0):
        o=observation();o['player'].update(hunger=hunger,thirst=80.);o['inventory']['berries']=food
        p=o['player'];o['nearby']=[dict(id='food',kind='berry',x=p['x']+dx,y=p['y'],available=True,ready_in=0)]
        o['allowed_actions']=[{'verb':'wait'},{'verb':'navigate','target_id':'food'}]
        return o

    def test_hungry_empty_agent_gets_more_food_potential_nearer_known_food(self):
        api=self.api();near=self.food_state(2);far=self.food_state(7)
        self.assertGreater(api.food_potential(near),api.food_potential(far))
        self.assertGreater(api.food_potential(self.food_state(7,food=3)),api.food_potential(far))
        unavailable=copy.deepcopy(near);unavailable['nearby'][0]['available']=False
        self.assertLess(api.food_potential(unavailable),api.food_potential(near))

    def test_eating_transfers_stored_nutrition_and_hoarding_saturates(self):
        api=self.api();before=self.food_state(hunger=20,food=2);after=self.food_state(hunger=38,food=1)
        self.assertAlmostEqual(api.food_potential(before),api.food_potential(after))
        self.assertEqual(api.food_potential(self.food_state(food=20)),api.food_potential(self.food_state(food=200)))
        self.assertEqual(api.food_potential(before,dead=True),0.)

    def test_food_shaping_telescopes_even_when_returning_to_same_state(self):
        api=self.api();a=self.food_state(7);b=self.food_state(2);b['tick']=a['tick']+1
        c=copy.deepcopy(a);c['tick']=b['tick']+1
        new=api.SupplyReward(a);old=OriginalReward(a);gamma=.9995
        diffs=[]
        for x,y in [(a,b),(b,c)]:
            n=new.score(step(x,y),y);o=old.score(step(x,y),y)
            diffs.append(n['reward']-o['reward']-n['components']['need_risk'])
        self.assertAlmostEqual(diffs[0]+gamma*diffs[1],(gamma**2-1)*api.food_potential(a),places=12)
        self.assertLess(diffs[0]+gamma*diffs[1],0.)

    def test_living_utility_stays_positive_at_zero_reserves_and_death_is_not_paid(self):
        api=self.api()
        self.assertGreater(api.living_utility({'hunger':0.,'thirst':0.,'dead':False}),0)
        self.assertLess(api.living_utility({'hunger':10.,'thirst':80.,'dead':False}),api.living_utility({'hunger':80.,'thirst':80.,'dead':False}))
        self.assertEqual(api.living_utility({'hunger':80.,'thirst':80.,'dead':True}),0)
        with self.assertRaises(ValueError):api.living_utility({'hunger':float('nan'),'thirst':80.,'dead':False})

    def test_reward_rejects_incomplete_trace_and_terminal_continuation(self):
        api=self.api();a=self.food_state();a['player']['health']=.1;b=copy.deepcopy(a)
        b['tick']=a['tick']+1;b['player']['health']=0.;b['allowed_actions']=[]
        tracker=api.SupplyReward(a);t=step(a,b,True)
        with self.assertRaises(ValueError):tracker.score(dict(t,trace=[]),b)
        reward=tracker.score(t,b)
        self.assertEqual(reward['components']['survival'],0.)
        self.assertEqual(reward['components']['need_risk'],0.)
        with self.assertRaises(ValueError):tracker.score(t,b)

    def test_reward_and_extra_critic_ignore_hidden_or_recommended_actions(self):
        api=self.api();o=self.food_state();altered=copy.deepcopy(o)
        altered.update(reward=999,teacher_action={'verb':'eat'},hidden_food={'x':0,'y':0})
        for x in altered['nearby']:x.update(path_steps=0,recommended=True)
        self.assertEqual(api.food_potential(o),api.food_potential(altered))
        self.assertEqual(api.readiness_features(encode(o)),api.readiness_features(encode(altered)))
        self.assertEqual(encode(o)['messages'],encode(altered)['messages'])

    def test_critic_readiness_distinguishes_regrowth_without_changing_actor_contract(self):
        api=self.api();a=self.food_state();b=copy.deepcopy(a);b['nearby'][0].update(available=False,ready_in=70)
        fa,fb=api.readiness_features(encode(a)),api.readiness_features(encode(b))
        self.assertEqual(len(fa),12);self.assertNotEqual(fa,fb)
        self.assertEqual(len(encode(a)['features']),96)

    def test_logical_batch16_uses_distinct_persistent_slots(self):
        from agent.m8s3_training import training_slots
        self.assertEqual(training_slots(1,16,32),list(range(16)))
        self.assertEqual(training_slots(2,16,32),list(range(16,32)))
        self.assertEqual(training_slots(3,16,32),list(range(16)))
        with self.assertRaises(ValueError):training_slots(1,12,32)

    def test_training_cli_accepts_registered_logical_batch_and_rejects_invalid_batch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from agent.m8s3_run import main
        template=json.loads((Path(__file__).resolve().parents[1]/'configs/m8s3_supply_pilot_control.json').read_text(encoding='utf-8'))
        for batch in (4,16,12):
            with tempfile.TemporaryDirectory() as tmp:
                config={**template,'project':tmp,'output':str(Path(tmp)/'run'),'envs':batch}
                registration=Path(tmp)/'registration.json'
                registration.write_text(json.dumps({'config':config}),encoding='utf-8')
                with patch('sys.argv',['m8s3_run','--registration',str(registration)]), \
                     patch('agent.m8s3_diag_contract.verify_registration_digest'), \
                     patch('agent.m8s3_run.verify_files'), patch('tools.benchmark_m8s3.setup'), \
                     patch('agent.m8s3_run.train') as train:
                    if batch==12:
                        with self.assertRaises(ValueError):main()
                        train.assert_not_called()
                    else:
                        main()
                        self.assertEqual(train.call_count,1)
                        self.assertEqual(train.call_args.args[0].envs,batch)

    def test_new_training_and_auditor_choose_the_same_explicit_reward(self):
        from agent.m8s3_run import training_reward
        from tools.audit_m8s3_training import replay_reward
        a=self.food_state();b=copy.deepcopy(a);b['tick']+=1
        for variant in ('legacy','supply_margin_v2'):
            config={'reward_variant':variant}
            self.assertEqual(training_reward(a,config).score(step(a,b),b),replay_reward(a,config).score(step(a,b),b))
        with self.assertRaises(ValueError):training_reward(a,{'reward_variant':'typo'})

    def test_registration_binds_reward_and_critic_dependencies(self):
        from pathlib import Path
        from tools.register_m8s3_training import registered_sources
        sources=registered_sources(Path(__file__).resolve().parents[1],{})
        self.assertIn('agent/m8s3_supply.py',sources)
        self.assertIn('tests/test_m8s3_supply.py',sources)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Node runtime supplies torch')
    def test_critic_expansion_preserves_all_old_predictions_then_can_learn_new_facts(self):
        import torch
        api=self.api();torch.manual_seed(3)
        critic=torch.nn.Sequential(torch.nn.Linear(96,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
        facts=torch.randn(5,96);expected=critic(facts).detach().clone()
        api.expand_critic(critic)
        extended=torch.cat([facts,torch.randn(5,12)],dim=1)
        self.assertTrue(torch.allclose(critic(extended),expected,atol=1e-6,rtol=0))
        critic(extended).square().mean().backward()
        self.assertGreater(float(critic[0].weight.grad[:,96:].abs().sum()),0.)


if __name__=='__main__':unittest.main()
