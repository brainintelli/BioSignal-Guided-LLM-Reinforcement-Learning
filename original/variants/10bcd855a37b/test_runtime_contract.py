"""CPU-only inspection of configuration and complete pairing/analysis contracts."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import sr_protocol as p
from sr_data import sha,read

ROOT=Path(__file__).parent


def function(name,filename,environment):
    tree=ast.parse((ROOT/filename).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(environment);exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    return namespace[name]


class RuntimeContractTests(unittest.TestCase):
    def test_all_twenty_four_configs_lock_r1_and_disable_r2(self):
        cfg=function('config','sr_common.py',dict(ARMS=p.ARMS,STATE_ARMS=p.STATE_ARMS,REWARD_ARMS=p.REWARD_ARMS,PROTOCOL=p.PROTOCOL,
            b=SimpleNamespace(CONFIG=dict(alpha=0.,rng_seed=1,r2_enabled=True,state2_enabled=True))))
        for block in range(8):
            for arm in p.ARMS:
                c=cfg(arm,block)
                self.assertNotIn('alpha',c);self.assertEqual(c['r1_weight'],1.)
                self.assertEqual(c['aux_weight'],0.);self.assertFalse(c['aux_enabled']);self.assertFalse(c['r2_enabled'])
                self.assertEqual(c['reward_contract'],'r1_only_v1')
                self.assertEqual(c['state2_enabled'],arm!='baseline')

    def test_pre_post_share_world_and_action_seeds_and_no_holdout(self):
        schedule=function('schedule','sr_run.py',dict(PROTOCOL=p.PROTOCOL,assignment=p.assignment))
        for block in range(8):
            standard=schedule(block,'baseline')
            for arm in p.ARMS:
                plans=schedule(block,arm)
                self.assertEqual([q['name'] for q in plans],['pre0','pre1','train','post0','post1'])
                self.assertEqual(len(plans[2]['order']),129)
                for j in range(2):
                    pre,post=plans[j],plans[3+j]
                    self.assertEqual(pre['world_seed'],post['world_seed'])
                    self.assertEqual(pre['action_seed'],post['action_seed'])
                    self.assertEqual(pre['order'],post['order'])
                    self.assertEqual(len(pre['order']),97)
                for q,ref in zip(plans,standard):
                    self.assertEqual((q['world_seed'],q['action_seed']),(ref['world_seed'],ref['action_seed']))
                    self.assertLess(max(q['order']),286)

    def test_twenty_four_cells_balanced_order_and_incomplete_analysis_blocks(self):
        cell_plan=function('cell_plan','sr_driver.py',dict(ARMS=p.ARMS,PROTOCOL=p.PROTOCOL))
        reg=dict(pilot=False,protocol=p.PROTOCOL,arm_orders={str(b):list(p.ARMS[b%3:]+p.ARMS[:b%3]) for b in range(8)})
        plan=cell_plan(reg)
        self.assertEqual(sum(len(v) for v in plan.values()),24)
        cells=[dict(block=b,arm=a,verified=True,pre=[{'discounted_r1':0.}]*2,post=[{'discounted_r1':float(b)}]*2) for b in range(8) for a in p.ARMS]
        self.assertEqual(p.analyze(cells[:-1])['status'],'incomplete')
        complete=p.analyze(cells)
        self.assertEqual(complete['status'],'complete')
        self.assertEqual(set(complete['primary']),{'state-baseline','state-shuffled'})
        self.assertIsNone(complete['biological_replication_units'])

    def test_trusted_runtime_helpers_are_byte_identical_to_previous_run(self):
        pins=read(ROOT/'legacy_source_pins.json')
        for name in ['base_smoke.py','host_cache.py','layout_offload.py','batch_boundary.py','sr_rollout.py','sr_verify.py','sr_analyze.py']:
            self.assertEqual(sha(ROOT/name),pins[name],name)

    def test_no_device_or_network_imports_in_new_data_boundary(self):
        for name in ['sr_data.py','export_spikeview.py','launch.py']:
            tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
            imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            imports += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
            self.assertFalse(any(x and x.split('.')[0] in ('maxlab','paramiko','requests','socket') for x in imports))


if __name__=='__main__':unittest.main()
