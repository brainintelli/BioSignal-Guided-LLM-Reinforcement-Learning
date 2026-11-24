import copy,math,unittest
from sr_protocol import PROTOCOL,ARMS,assignment,reward_record,fit_profile,score,features,episode_metrics,paired_stats,analyze

def synthetic():
    return [dict(index=i,spike_count=100+i%11,rate_hz_per_channel=(100+i%11)/850,active_channel_fraction=.1+(i%7)/100,
        mean_abs_amplitude_uv=30+(i%13),coincident_event_fraction=.8+(i%5)/100) for i in range(298)]

class ProtocolTests(unittest.TestCase):
    def test_separated_finite_split_and_block_shuffle(self):
        train=assignment('both','train',0);ev=assignment('both','eval',0)
        self.assertEqual(train,list(range(60,189)));self.assertEqual(ev,list(range(199,296)))
        self.assertFalse(set(train)&set(ev));self.assertFalse(set(train+ev)&set(range(60)))
        for split,base in [('train',train),('eval',ev)]:
            variants=[]
            for block in range(8):
                order=assignment('shuffled',split,block);self.assertEqual(order[-1],base[-1]);self.assertEqual(sorted(order),base)
                self.assertTrue(all(a!=b for a,b in zip(order[:-1],base[:-1])));self.assertEqual(order,assignment('shuffled',split,block));variants.append(tuple(order))
            self.assertEqual(len(set(variants)),8)
        with self.assertRaises(ValueError):assignment('baseline','train',8)
    def test_fit_never_reads_training_or_eval_features(self):
        x=synthetic();a=fit_profile(x);y=copy.deepcopy(x)
        for w in y[60:]:w.update(spike_count=1e8,mean_abs_amplitude_uv=1e8,rate_hz_per_channel=1e8)
        self.assertEqual(a,fit_profile(y));self.assertAlmostEqual(sum(score(w,a)['u'] for w in x[:60]),0,places=12)
        self.assertTrue(all(-1<=score(w,a)['u']<=1 for w in x));self.assertTrue(all(abs(v)<=5 for v in features(y[-1],a)))
    def test_score_ignores_task_action_and_reward(self):
        x=synthetic();p=fit_profile(x);a=score(x[60],p);b=score(x[60]|dict(action='eat',r1=1e9),p);self.assertEqual(a,b)
        self.assertLessEqual(abs(reward_record(.4,a,'both',training=True)['total_reward']-.4),.05)
    def test_reward_flags_and_unscaled_r1(self):
        s=dict(p=.7,u=-.5)
        for arm in ARMS:
            r=reward_record(.3,s,arm,training=True);active=arm in ('reward','both','shuffled')
            self.assertEqual(r['r1_weight'],1.);self.assertEqual(r['aux_weight'],.05 if active else 0.)
            self.assertEqual(r['total_reward'],.3+(.05 if active else 0.)*(-.5));self.assertFalse(r['biological_validated'])
            self.assertEqual(reward_record(.3,s,arm,training=False)['total_reward'],.3)
        with self.assertRaises(ValueError):reward_record(float('nan'),s,'both',training=True)
    def test_shuffling_preserves_state_reward_marginals(self):
        x=synthetic();p=fit_profile(x)
        for split in ('train','eval'):
            a=assignment('both',split,4)[:-1];b=assignment('shuffled',split,4)[:-1]
            self.assertEqual(sorted(tuple(features(x[i],p))+tuple(score(x[i],p).values()) for i in a),sorted(tuple(features(x[i],p))+tuple(score(x[i],p).values()) for i in b))
    def test_metrics_use_r1_and_variable_duration(self):
        rows=[dict(r1=1.,duration=2,terminal=False),dict(r1=2.,duration=3,terminal=True)]
        m=episode_metrics(rows);self.assertAlmostEqual(m['discounted_r1'],1+PROTOCOL['gamma']**2*2);self.assertEqual(m['ticks'],5)
    def test_exact_null_and_positive_pair_statistics(self):
        z=paired_stats([0.]*8);self.assertEqual(z['p_two_sided'],1.);self.assertEqual(z['bootstrap_95'],[0.,0.])
        a=paired_stats([1.]*8);self.assertEqual(a['p_two_sided'],2/256);self.assertEqual(a['p_adjusted_primary'],4/256)
    def test_analysis_rejects_missing_duplicates_nonfinite(self):
        cells=[dict(block=b,arm=a,pre=[dict(discounted_r1=0.)]*2,post=[dict(discounted_r1=float(a=='both'))]*2,verified=True) for b in range(8) for a in ARMS]
        self.assertEqual(analyze(cells[:-1])['status'],'incomplete');self.assertEqual(analyze(cells)['status'],'complete')
        with self.assertRaises(ValueError):analyze(cells+[cells[0]])
        bad=copy.deepcopy(cells);bad[0]['post'][0]['discounted_r1']=float('nan')
        with self.assertRaises(ValueError):analyze(bad)
        bad=copy.deepcopy(cells);bad[0]['verified']=False
        with self.assertRaises(ValueError):analyze(bad)

if __name__=='__main__':unittest.main()
