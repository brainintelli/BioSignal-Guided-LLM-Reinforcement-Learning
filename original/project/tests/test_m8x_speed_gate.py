"""A speed claim requires equal real decisions and actual optimizer work."""
import importlib
import importlib.util
import unittest

class SpeedGateTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.benchmark_m8x_kv'))
        return importlib.import_module('tools.benchmark_m8x_kv')

    def rows(self,seconds,steps=8):
        return [dict(decisions=128,optimizer_steps=steps,rollout_seconds=seconds/4,
                     update_seconds=seconds*3/4,finite=True,zero_update_logp_max=0.) for _ in range(3)]

    def test_exactly_two_is_not_more_than_two(self):
        gate=self.module().speed_gate
        self.assertFalse(gate(self.rows(100),self.rows(50))['passed'])
        self.assertTrue(gate(self.rows(100),self.rows(49))['passed'])

    def test_fewer_optimizer_steps_cannot_masquerade_as_speedup(self):
        gate=self.module().speed_gate
        self.assertFalse(gate(self.rows(100),self.rows(30,steps=4))['passed'])
        rows=self.rows(30);rows[0]['decisions']=64
        self.assertFalse(gate(self.rows(100),rows)['passed'])

    def test_cold_cost_and_probability_drift_cannot_be_hidden(self):
        gate=self.module().speed_gate
        rows=self.rows(30);rows[0]['update_seconds']=200
        self.assertFalse(gate(self.rows(100),rows)['passed'])
        rows=self.rows(30);rows[1]['zero_update_logp_max']=1e-4
        self.assertFalse(gate(self.rows(100),rows)['passed'])

if __name__=='__main__':unittest.main()
