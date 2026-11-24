"""Contract/trie acceptance tests before connecting the semantic GPU policy."""
import importlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class SemanticContractTests(unittest.TestCase):
    def module(self,name):
        self.assertIsNotNone(importlib.util.find_spec(name),'Missing '+name)
        return importlib.import_module(name)

    def test_true_record_preserves_actions_and_critic_information(self):
        m=self.module('agent.m8s_contract')
        from agent.m8x_contract import encode,critic_features
        records=[json.loads(x) for x in (ROOT/'output/rl_kv2_20261107/optimized_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
        for record in records:
            old=encode(record['observation']);new=m.encode(record['observation'])
            self.assertEqual(new['branches'],old['branches'])
            self.assertEqual(new['features'],critic_features(old))
            self.assertNotIn('open3',new['messages'][1]['content'])
            self.assertEqual(new['verb_mask'],old['verb_mask'])
            for verb,branch in enumerate(new['branches']):
                self.assertEqual(len(new['target_texts'][verb]),len(branch))
                self.assertEqual(len(set(new['target_texts'][verb])),len(branch))
                if verb==0:
                    self.assertTrue(set(new['target_texts'][verb])<=set(('向上','向右','向下','向左')))

    def test_known_target_text_and_no_letter_instructions(self):
        m=self.module('agent.m8s_contract')
        r=json.loads((ROOT/'output/rl_kv2_20261107/real_example.json').read_text(encoding='utf-8'))
        new=m.encode(r['observation'])
        self.assertEqual(new['target_texts'][1][1],'树木（左3格、上1格）')
        self.assertIn('距入夜140刻',new['messages'][1]['content'])
        self.assertNotIn('目标字母',m.target_messages(new,1)[-1]['content'])
        self.assertEqual(m.CODES,('移动','前往','采集','吃饭','喝水','休息','建造','等待'))

    def test_trie_decisions_skip_unary_paths_without_losing_leaves(self):
        m=self.module('agent.m8s_trie')
        # Include a candidate that is a prefix of another: EOS must disambiguate.
        seqs=[[10,20],[10,21],[11],[11,30]]
        nodes,paths=m.choice_tree(seqs,eos=99)
        self.assertEqual(nodes,{():[10,11],(10,):[20,21],(11,):[30,99]})
        self.assertEqual(len(paths),4)
        import math
        def softmax(xs):
            es=[math.exp(x-max(xs)) for x in xs];return [x/sum(es) for x in es]
        ps={prefix:softmax([float(i) for i in range(len(tokens))]) for prefix,tokens in nodes.items()}
        probabilities=[math.prod(ps[prefix][index] for prefix,index in path) for path in paths]
        self.assertAlmostEqual(sum(probabilities),1.)
        self.assertEqual(paths[0],[((),0),((10,),0)])

    def test_duplicate_and_empty_targets_fail_closed(self):
        m=self.module('agent.m8s_trie')
        for seqs in ([[1],[1]],[[]],[]):
            with self.assertRaises(ValueError):m.choice_tree(seqs,eos=99)

    def test_tree_label_permutation_only_permutes_leaf_paths(self):
        m=self.module('agent.m8s_trie')
        a=[[4,1,8],[4,1,9],[4,2],[7,3]]
        nodes,paths=m.choice_tree(a,eos=99)
        nodes2,paths2=m.choice_tree(a[::-1],eos=99)
        self.assertEqual(nodes,nodes2)
        self.assertEqual(paths,paths2[::-1])

if __name__=='__main__':unittest.main(verbosity=2)
