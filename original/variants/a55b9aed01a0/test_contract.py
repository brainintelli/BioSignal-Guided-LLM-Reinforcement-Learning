import copy,json,unittest
from pathlib import Path
from sr_data import Data,sha,digest
from sr_contract import encode,context_id
from agent import m9_contract
R=Path(__file__).resolve().parent
class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.obs=json.loads((R/'contract_fixture.json').read_text(encoding='utf-8'));cls.data=Data(R/'dataset.json',sha(R/'dataset.json'))
    def test_state_and_reward_only_inputs_are_matched(self):
        s=self.data.snapshot(60,context_id(self.obs));base=m9_contract.encode(self.obs,'reserves')
        a=encode(self.obs,s,'baseline');b=encode(self.obs,s,'reward')
        self.assertEqual(a['messages'],base['messages']);self.assertEqual(a['messages'],b['messages']);self.assertEqual(a['critic_input'],b['critic_input'])
        self.assertEqual(len(a['critic_input']),112);self.assertEqual(a['critic_input'][-4:],[0.]*4)
        c=encode(self.obs,s,'state');d=encode(self.obs,s,'both');e=encode(self.obs,s,'shuffled')
        self.assertEqual(c['messages'],d['messages']);self.assertEqual(c['messages'],e['messages']);self.assertEqual(c['critic_input'][-4:],s['state_values'])
        self.assertNotEqual(c['messages'],a['messages']);self.assertEqual(c['branches'],a['branches'])
    def test_no_arm_window_score_or_future_labels_in_text(self):
        s=self.data.snapshot(60,context_id(self.obs));a=encode(self.obs,s,'both')
        text=a['messages'][1]['content'];self.assertNotIn('both',text);self.assertNotIn('snapshot_id',text);self.assertNotIn('aux_probability',text)
        self.assertIn('离线',text);self.assertIn('无因果对应',text)
    def test_snapshot_tampering_or_wrong_context_fails(self):
        s=self.data.snapshot(60,context_id(self.obs));bad=copy.deepcopy(s);bad['state_values'][0]+=.1
        with self.assertRaises(ValueError):encode(self.obs,bad,'both')
        bad=self.data.snapshot(60,'wrong')
        with self.assertRaises(ValueError):encode(self.obs,bad,'both')
        bad=copy.deepcopy(s);bad['source_full_qc_passed']=True;bad['snapshot_id']=digest({k:v for k,v in bad.items() if k!='snapshot_id'})
        with self.assertRaises(ValueError):encode(self.obs,bad,'both')
if __name__=='__main__':unittest.main()
