"""CPU online identity/caching/sampling checks, never real CUDA evidence."""
import copy
import unittest

from agent import m8x_contract as contract, m8x_server
from agent.m6_live_session import SessionRegistry
from agent.rl_contract import LABELS
from tests.test_m8x_contract import observation
from tests.m8x_online_fixtures import FakeTorch, FakePolicy, Stream


def record_fixture():
    obs=observation();obs['tick']=0
    row=contract.encode(obs);verb=next(i for i,b in enumerate(row['branches']) if b)
    decision=dict(m8x_server.decision_identity('a'*64,'b'*64),tick=0,source='model',raw_valid=True,
        action=row['branches'][verb][0],verb_index=verb,target_index=0,model_input=row['messages'],
        target_input=contract.target_messages(row,verb) if len(row['branches'][verb])>1 else None,
        verb_probabilities=[float(i==verb) for i in range(len(contract.CODES))],
        target_probabilities=[float(i==0) for i in range(len(LABELS))],
        raw_output=contract.CODES[verb]+':'+m8x_server.target_code(row,verb,0))
    log=[];registry=SessionRegistry(Stream,log.append)
    context=dict(session_id='balanced-synthetic-fixture',episode=0,world_seed=42)
    def sample(stream):stream.value+=2;return decision
    registry.decide(context,obs,sample)
    return log[0],context,obs


class ExplorationOnlineTests(unittest.TestCase):
    def test_record_has_balanced_world_and_original_letter_targets(self):
        record,_,_=record_fixture();decision=record['decision']
        m8x_server.validate_log_identity(record,'a'*64,'b'*64)
        for key,value in contract.balance_identity().items():self.assertEqual(decision[key],value)
        self.assertEqual(decision['observation_format'],contract.VERSION)
        row=contract.encode(observation())
        for verb,branch in enumerate(row['branches']):
            for target in range(len(branch)):
                self.assertEqual(m8x_server.target_code(row,verb,target),LABELS[target])

    def test_world_actor_contract_and_input_corruption_rejected(self):
        record,_,_=record_fixture()
        bad=[('profile','open1'),('world_version',2),('environment_version','old'),
             ('balance_sha256','0'*64),('actor_sha256','c'*64),('observation_format','v4'),
             ('physical_forward_shape',[1,1536]),('source','fallback'),('model_input',[]),
             ('raw_output','wait:A'),('target_index',63),('verb_probabilities',[0.]*8)]
        for key,value in bad:
            changed=copy.deepcopy(record);changed['decision'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m8x_server.validate_log_identity(changed,'a'*64,'b'*64)
        changed=copy.deepcopy(record);changed['observation']['balance_sha256']='0'*64
        with self.assertRaises(ValueError):m8x_server.validate_log_identity(changed,'a'*64,'b'*64)

    def test_changed_legal_action_is_not_a_valid_cached_decision(self):
        record,_,obs=record_fixture()
        record['decision']['action']=next(a for a in obs['allowed_actions'] if a!=record['decision']['action'])
        with self.assertRaises(ValueError):m8x_server.validate_log_identity(record,'a'*64,'b'*64)

    def test_retry_restores_original_decision_without_sampling(self):
        record,context,obs=record_fixture()
        m8x_server.validate_log_identity(record,'a'*64,'b'*64)
        sessions=SessionRegistry(Stream,lambda _:self.fail('Duplicate log write'))
        sessions.restore_records([record])
        self.assertEqual(sessions.decide(context,obs,lambda _:self.fail('Duplicate sample')),record['decision'])

    def test_single_target_still_draws_twice_and_saved_rng_is_restored(self):
        torch=FakeTorch();stream=Stream(0)
        prepared=[dict(verb_mask=[True],branches=[[{'verb':'wait'}]])]
        result=m8x_server.sample_stochastic(FakePolicy(),prepared,stream,torch)
        self.assertEqual(len(torch.draws),2)
        self.assertEqual(result['action'],{'verb':'wait'})
        record,_,_=record_fixture();stream=Stream(0)
        m8x_server.verify_saved_cuda_draws(record,stream,FakeTorch())
        self.assertEqual(stream.value,0)


if __name__=='__main__':unittest.main()
