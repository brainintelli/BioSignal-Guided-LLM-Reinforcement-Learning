import tempfile,unittest,json
from pathlib import Path
from multipoint import MultiLedger,events_for

class LedgerTests(unittest.TestCase):
 def test_exact_20_by_15_plan_and_no_sham(self):
  e=[x for n in range(1,21) for x in events_for(n,1000+n,'session')]
  self.assertEqual(len(e),300);self.assertEqual(len({x['id'] for x in e}),300);self.assertEqual({x['arm'] for x in e},{'true'})
 def test_reserved_whole_train_lost_ack_is_not_reused(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'claims').mkdir();m=MultiLedger(r)
   e=events_for(1,1001,'session');m.reserve(e)
   self.assertEqual(m.total(),15)
   with self.assertRaises(FileExistsError):m.reserve(e)
   for n in range(2,21):m.reserve(events_for(n,1000+n,'session'))
   self.assertEqual(m.total(),300)
   with self.assertRaises(ValueError):m.reserve(events_for(21,1021,'session'))
 def test_prior_true_doses_count_toward_300(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'claims').mkdir();(r/'claims/1.json').write_text(json.dumps({'arm':'true'}));m=MultiLedger(r)
   for n in range(1,20):m.reserve(events_for(n,1000+n,'session'))
   with self.assertRaises(ValueError):m.reserve(events_for(20,1020,'session'))

if __name__=='__main__':unittest.main()
