import copy,unittest
from sr_driver import cell_plan
from sr_analyze import validate_receipt
from sr_protocol import ARMS,PROTOCOL
class DeliveryRulesTests(unittest.TestCase):
    def test_exact_registered_matrix_and_order(self):
        r=dict(pilot=False,protocol=PROTOCOL,arm_orders={str(b):list(ARMS[b%5:]+ARMS[:b%5]) for b in range(8)})
        p=cell_plan(r);self.assertEqual(sum(map(len,p.values())),40)
        for bad in [r|dict(pilot=True),r|dict(arm_orders={**r['arm_orders'],'2':['both']*5})]:
            with self.assertRaises(ValueError):cell_plan(bad)
    def test_receipt_is_bound_to_cell_manifest_registration_and_errors(self):
        r=dict(block=2,arm='both',collection_pid=11)
        v=dict(passed=True,block=2,arm='both',collection_pid=11,verification_pid=12,registration_sha256='reg',collection_manifest_sha256='man',
            actual_weight_optimizer_checks=True,updated_train_statistic_error=0.,phases=[dict(phase=n,statistic_error=0.,bootstrap_error=0.,native_replay=True,snapshot_inputs_verified=True,reward_gae_verified=True) for n in ['pre0','pre1','train','post0','post1']])
        self.assertTrue(validate_receipt(r,v,'reg','man'))
        for change in [dict(block=3),dict(collection_manifest_sha256='different'),dict(verification_pid=11),dict(passed=False),dict(updated_train_statistic_error=1e-4)]:
            with self.assertRaises(ValueError):validate_receipt(r,v|change,'reg','man')
        bad=copy.deepcopy(v);bad['phases'][0]['statistic_error']=float('nan')
        with self.assertRaises(ValueError):validate_receipt(r,bad,'reg','man')
if __name__=='__main__':unittest.main()
