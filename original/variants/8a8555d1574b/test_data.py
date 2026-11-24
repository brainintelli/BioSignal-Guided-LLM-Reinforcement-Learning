import copy,unittest
from sr_data import scoped_events,window_rows,digest

class DataTests(unittest.TestCase):
    def test_scope_rejects_unmapped_and_outside_but_keeps_exact_boundary(self):
        events=[[0,-1,0,-2.],[1,0,0,-2.],[2,20,9,1.],[3,99,1,3.],[4,100,1,2.]]
        valid,excluded=scoped_events(events,0,100,[0,1]);self.assertEqual([r[0] for r in valid],[1,3]);self.assertEqual(excluded,[0,2,4])
    def test_integral_windows_and_distinct_channel_coincidence(self):
        events=[[0,19999,0,-2.],[1,20000,0,-2.],[2,20001,0,-3.],[3,20100,1,4.],[4,40000,0,5.],[5,80000,0,1.]]
        w=window_rows(events,0,100000,[0,1],.000001)
        self.assertEqual(len(w),3);self.assertEqual(w[0]['start_frame'],20000);self.assertEqual(w[0]['end_frame'],40000)
        self.assertEqual(w[0]['spike_count'],3);self.assertEqual(w[0]['coincident_event_fraction'],1.);self.assertEqual(w[0]['mean_abs_amplitude_uv'],3.)
        self.assertEqual(w[1]['spike_count'],1);self.assertEqual(w[1]['coincident_event_fraction'],0.);self.assertEqual(w[2]['spike_count'],0)
        self.assertEqual(w[2]['mean_abs_amplitude_uv'],0.);self.assertNotEqual(w[0]['events_sha256'],w[1]['events_sha256'])
    def test_duplicates_nonfinite_or_unordered_fail(self):
        for events in ([[0,3,0,1.],[1,3,0,2.]],[[0,3,0,float('nan')]],[[0,4,0,1.],[1,3,0,1.]]):
            with self.assertRaises(ValueError):scoped_events(events,0,100,[0])
    def test_canonical_identity_changes_with_values(self):
        a=dict(x=[1,2],b='recording');self.assertEqual(digest(a),digest(dict(b='recording',x=[1,2])));b=copy.deepcopy(a);b['x'][0]=3;self.assertNotEqual(digest(a),digest(b))
if __name__=='__main__':unittest.main()
