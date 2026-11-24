import importlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'standalone/mea_state2_r2'))


class TerminalContextTests(unittest.TestCase):
    def helper(self):
        module=importlib.import_module('mea_state2_r2.features')
        self.assertTrue(hasattr(module,'transition_context'),'Terminal reward context must bypass policy action encoding')
        return module.transition_context,module.digest

    def test_dead_observation_never_enters_legal_action_encoder(self):
        helper,digest=self.helper();obs={'dead':True,'legal_actions':[],'tick':491}
        def forbidden(_):raise AssertionError('Dead state has no legal policy actions')
        self.assertEqual(helper(obs,True,forbidden),'terminal:'+digest(obs))

    def test_live_observation_keeps_existing_encoder_identity(self):
        helper,_=self.helper();obs={'dead':False,'tick':490};calls=[]
        def encoder(value):calls.append(value);return 'existing-live-id'
        self.assertEqual(helper(obs,False,encoder),'existing-live-id')
        self.assertEqual(calls,[obs])


if __name__=='__main__':unittest.main()
