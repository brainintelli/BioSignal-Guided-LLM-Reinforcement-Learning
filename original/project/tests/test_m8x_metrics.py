import unittest
from tools.m8x_metrics import qualify_selection,compare_test,CHECKPOINTS

def episodes(n,t=1800):
 return [dict(seed=i,replicate=0,survival_ticks=t,terminal=t<1800,alive_at_horizon=t==1800) for i in range(n)]
class BalancedStats(unittest.TestCase):
 def test_complete_checkpoint_set_and_parent_preference(self):
  es=episodes(16);r=qualify_selection({x:es for x in CHECKPOINTS},es)
  self.assertTrue(r['eligible_for_test']);self.assertEqual(r['selected_decisions'],32768)
  with self.assertRaises(ValueError):qualify_selection({32768:es},es)
 def test_all_success_not_population_proof_and_no_rl_gain(self):
  es=episodes(100);r=compare_test(es,es,episodes(100,250))
  self.assertTrue(r['numerical_gate_passed']);self.assertTrue(r['prefer_parent'])
  self.assertFalse(r['rl_improvement_demonstrated']);self.assertFalse(r['release_authorized'])
  self.assertLess(r['success_interval']['one_sided_95_lower'],1)
 def test_world_amendment_comparison_is_separate_from_rl_gain(self):
  r=compare_test(episodes(100),episodes(100),episodes(100,250))
  world=r['balanced_parent_vs_original_v2']
  self.assertEqual(world['mean_difference'],1550)
  self.assertEqual(world['paired_95ci'],[1550,1550])
  self.assertEqual(world['success_difference'],1)
  self.assertEqual(world['success_paired_95ci'],[1,1])
  self.assertFalse(r['rl_improvement_demonstrated'])
 def test_1800_death_is_failure(self):
  es=episodes(16);es[0].update(terminal=True,alive_at_horizon=False)
  r=qualify_selection({x:es for x in CHECKPOINTS},episodes(16))
  self.assertFalse(r['eligible_for_test'])
 def test_rmst_margin_is_strict_ci_not_point_gain(self):
  es=episodes(100,1710);r=compare_test(es,episodes(100),episodes(100,250))
  self.assertFalse(r['noninferiority']['rmst_passed'])
 def test_changed_world_and_rep_rejected(self):
  es=episodes(100);es[-1]['replicate']=1
  with self.assertRaises(ValueError):compare_test(es,episodes(100),episodes(100))
if __name__=='__main__':unittest.main()
