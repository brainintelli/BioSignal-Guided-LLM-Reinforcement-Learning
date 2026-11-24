import copy
import unittest

from mea_binary_r2.encoding import Encoder
try:
    from mea_binary_r2.contract import reward_record, TransitionGate, AttemptBudget
except ImportError:
    reward_record = TransitionGate = AttemptBudget = None


IDENTITY=dict(session_id='session-1',chip_id='test-chip',organoid_id='test-organoid',
              route_id='route-1',episode_id='episode-1',transition_id='step-1',context_id='after-action-1')


def stimulus(pulses=1):
    return Encoder().decide({'H':100,'F':100,'W':40 if pulses==3 else 80})


def quiet():
    return Encoder().decide({'H':100,'F':100,'W':100})


def evidence(pulses, identity=None, anchor=100000, event_id='event-1', window_id='window-1'):
    return dict(source='simulation',identity=copy.deepcopy(identity or IDENTITY),
                event_id=event_id,window_id=window_id,status='confirmed',sample_rate_hz=20000,
                anchor_frame=anchor,pulse_frames=[anchor+i*6000 for i in range(pulses)],
                max_abs_output_mv=298.7 if pulses else 0.,output_verified=True,route_verified=True,
                window_start_frame=anchor+240000,window_end_frame=anchor+260000,
                coverage_start_frame=anchor-60000,coverage_end_frame=anchor+260000,
                complete=True,dropped_frames=0,qc_passed=True)


def kwargs(intent=None,receipt=None,**overrides):
    intent=intent or stimulus()
    row=dict(r1=.25,duration=2,gamma=.5,beta=.002,intent=intent,
             evidence=evidence(intent.pulses) if receipt is None else receipt,
             expected_identity=copy.deepcopy(IDENTITY))
    row.update(overrides)
    return row


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(reward_record,'binary R2 contract has not been implemented')

    def test_quiet_positive_and_both_stimulus_counts_negative(self):
        for intent,raw,total in [(quiet(),1,.253),(stimulus(),-1,.247),(stimulus(3),-1,.247)]:
            with self.subTest(pulses=intent.pulses):
                result=reward_record(**kwargs(intent))
                self.assertEqual(result['r2_raw'],raw)
                self.assertEqual(result['r2_train'],raw*1.5)
                self.assertAlmostEqual(result['total_reward'],total)
                self.assertTrue(result['valid'])
                self.assertEqual(result['source'],'simulation')

    def test_unknown_output_is_never_positive(self):
        for field,value in [('status','unknown'),('output_verified',False),('route_verified',False)]:
            receipt=evidence(0);receipt[field]=value
            result=reward_record(**kwargs(quiet(),receipt))
            self.assertFalse(result['valid'])
            self.assertIsNone(result['r2_raw'])
            self.assertIsNone(result['total_reward'])

    def test_recovery_failure_keeps_audit_negative_but_blocks_training(self):
        receipt=evidence(1);receipt['qc_passed']=False
        result=reward_record(**kwargs(receipt=receipt))
        self.assertEqual(result['r2_raw'],-1)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['total_reward'])

    def test_planned_stimulus_not_observed_cannot_earn_quiet_bonus(self):
        result=reward_record(**kwargs(receipt=evidence(0)))
        self.assertFalse(result['valid'])
        self.assertIsNone(result['r2_raw'])
        self.assertIsNone(result['total_reward'])

    def test_unrequested_stimulus_retains_negative_audit_and_is_invalid(self):
        result=reward_record(**kwargs(quiet(),evidence(1)))
        self.assertEqual(result['r2_raw'],-1)
        self.assertFalse(result['valid'])

    def test_missing_coverage_and_wrong_timing_cannot_train(self):
        for field,value in [('coverage_end_frame',359999),('coverage_start_frame',100001),
                            ('window_start_frame',340001),('window_end_frame',359999),
                            ('complete',False),('dropped_frames',1),
                            ('pulse_frames',[100001]),('max_abs_output_mv',301.)]:
            with self.subTest(field=field):
                receipt=evidence(1);receipt[field]=value
                result=reward_record(**kwargs(receipt=receipt))
                self.assertFalse(result['valid'])
                self.assertIsNone(result['total_reward'])

    def test_cross_identity_and_live_provenance_rejected(self):
        for key in IDENTITY:
            receipt=evidence(1);receipt['identity'][key]='foreign'
            with self.subTest(key=key),self.assertRaises(ValueError):
                reward_record(**kwargs(receipt=receipt))
        for source in ('live','recording',''):
            receipt=evidence(1);receipt['source']=source
            with self.subTest(source=source),self.assertRaises(ValueError):
                reward_record(**kwargs(receipt=receipt))

    def test_bad_types_numbers_schema_and_receipt_mutation(self):
        for key,value in [('r1',True),('r1',float('nan')),('duration',True),('duration',0),
                          ('duration',10001),('gamma',1),('gamma',0),('beta',-.1),('beta',True)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                reward_record(**kwargs(**{key:value}))
        for field,value in [('qc_passed',1),('output_verified',1),('complete',1),
                            ('pulse_frames',[True]),('sample_rate_hz',10000),
                            ('max_abs_output_mv',float('inf'))]:
            receipt=evidence(1);receipt[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                reward_record(**kwargs(receipt=receipt))
        receipt=evidence(1);receipt['unexpected']=True
        with self.assertRaises(ValueError):reward_record(**kwargs(receipt=receipt))
        receipt=evidence(1);saved=copy.deepcopy(receipt)
        result=reward_record(**kwargs(receipt=receipt));result['identity']['chip_id']='changed'
        self.assertEqual(receipt,saved)

    def test_terminal_retains_r1_without_stimulation_or_positive_bonus(self):
        intent=Encoder().decide({'H':0,'F':0,'W':0},terminal=True)
        result=reward_record(**kwargs(intent,evidence=None,terminal=True))
        self.assertEqual(result['total_reward'],.25)
        self.assertIsNone(result['r2_raw'])
        self.assertEqual(result['r2_train'],0.)
        self.assertFalse(result['eligible'])
        self.assertTrue(result['valid'])
        with self.assertRaises(ValueError):reward_record(**kwargs(quiet(),terminal=True))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TransitionGate,'transaction gate has not been implemented')
        self.binding={k:IDENTITY[k] for k in ('session_id','chip_id','organoid_id','route_id')}

    def test_one_response_per_transition_and_new_frames(self):
        gate=TransitionGate(self.binding)
        self.assertTrue(gate.accept(**kwargs())['valid'])
        next_id=IDENTITY|{'transition_id':'step-2','context_id':'after-action-2'}
        result=gate.accept(**kwargs(receipt=evidence(1,next_id,400000,'event-2','window-2'),expected_identity=next_id))
        self.assertTrue(result['valid'])
        with self.assertRaises(ValueError):gate.accept(**kwargs())
        self.assertTrue(gate.stopped)

    def test_duplicate_window_event_or_overlapping_frame_stops_session(self):
        for receipt in (evidence(1,anchor=400000,event_id='event-2'),
                        evidence(1,anchor=400000,window_id='window-2'),
                        evidence(1,anchor=200000,event_id='event-2',window_id='window-2')):
            gate=TransitionGate(self.binding);gate.accept(**kwargs())
            next_id=IDENTITY|{'transition_id':'step-2','context_id':'after-action-2'}
            receipt['identity']=next_id
            with self.subTest(receipt=receipt),self.assertRaises(ValueError):
                gate.accept(**kwargs(receipt=receipt,expected_identity=next_id))
            self.assertTrue(gate.stopped)

    def test_invalid_result_cannot_be_retried_as_quiet(self):
        gate=TransitionGate(self.binding)
        receipt=evidence(1);receipt['status']='unknown'
        self.assertFalse(gate.accept(**kwargs(receipt=receipt))['valid'])
        with self.assertRaises(ValueError):gate.accept(**kwargs(quiet()))

    def test_binding_is_owned_and_foreign_session_stops(self):
        gate=TransitionGate(self.binding);self.binding['chip_id']='mutated'
        self.assertTrue(gate.accept(**kwargs())['valid'])
        other=IDENTITY|{'chip_id':'foreign','transition_id':'step-2'}
        with self.assertRaises(ValueError):
            gate.accept(**kwargs(receipt=evidence(1,other,400000,'event-2','window-2'),expected_identity=other))

    def test_terminal_episode_cannot_resume_but_new_episode_can(self):
        dead=Encoder().decide({'H':0,'F':0,'W':0},terminal=True)
        for episode,allowed in [('episode-1',False),('episode-2',True)]:
            gate=TransitionGate(self.binding)
            gate.accept(**kwargs(dead,evidence=None,terminal=True))
            next_id=IDENTITY|{'episode_id':episode,'transition_id':'after-death',
                             'context_id':'new-game-observation'}
            call=kwargs(quiet(),evidence(0,next_id,400000,'event-2','window-2'),expected_identity=next_id)
            with self.subTest(episode=episode):
                if allowed:
                    result=gate.accept(**call)
                    self.assertTrue(result['valid'])
                    self.assertEqual(result['r2_raw'],1)
                else:
                    with self.assertRaises(ValueError):gate.accept(**call)
                    self.assertTrue(gate.stopped)


class Clock:
    def __init__(self):self.wall_value=1000.;self.mono_value=10.
    def time(self):return self.wall_value
    def monotonic(self):return self.mono_value
    def advance(self,seconds):self.wall_value+=seconds;self.mono_value+=seconds


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(AttemptBudget,'attempt budget has not been implemented')
        self.clock=Clock()

    def test_count_before_output_and_unknown_never_retried(self):
        budget=AttemptBudget(1000.,self.clock)
        budget.admit(stimulus(3))
        self.assertEqual((budget.events,budget.pulses),(1,3))
        with self.assertRaises(ValueError):budget.admit(stimulus())
        budget=AttemptBudget(1000.,self.clock)
        budget.admit(stimulus());budget.complete(False)
        with self.assertRaises(ValueError):budget.admit(quiet())

    def test_event_limit_also_blocks_quiet_reward_after_exhaustion(self):
        budget=AttemptBudget(1000.,self.clock)
        for _ in range(30):
            budget.admit(stimulus(3));budget.complete(True);self.clock.advance(20)
        self.assertEqual((budget.events,budget.pulses),(30,90))
        with self.assertRaises(ValueError):budget.admit(quiet())

    def test_output_slots_cannot_be_faster_than_observation_cycle(self):
        budget=AttemptBudget(1000.,self.clock)
        budget.admit(quiet());budget.complete(True)
        self.clock.advance(19.999)
        with self.assertRaises(ValueError):budget.admit(stimulus())
        self.assertTrue(budget.stopped)

    def test_original_start_and_recovery_reserve(self):
        self.clock.advance(5680)
        budget=AttemptBudget(1000.,self.clock)
        budget.admit(stimulus());budget.complete(True)
        self.clock.advance(.001)
        with self.assertRaises(ValueError):budget.admit(stimulus())
        self.clock.advance(1520)
        with self.assertRaises(ValueError):AttemptBudget(1000.,self.clock)

    def test_wall_or_monotonic_regression_stops_session(self):
        for attribute in ('wall_value','mono_value'):
            clock=Clock();budget=AttemptBudget(1000.,clock)
            setattr(clock,attribute,getattr(clock,attribute)-1)
            with self.subTest(attribute=attribute),self.assertRaises(ValueError):budget.admit(quiet())
            clock.advance(10)
            with self.assertRaises(ValueError):budget.admit(stimulus())

    def test_terminal_and_invalid_clock_do_not_create_attempts(self):
        budget=AttemptBudget(1000.,self.clock)
        intent=Encoder().decide({'H':0,'F':0,'W':0},terminal=True)
        with self.assertRaises(ValueError):budget.admit(intent)
        self.assertEqual((budget.events,budget.pulses),(0,0))
        for start in (True,float('nan'),1001.,-6201.):
            with self.subTest(start=start),self.assertRaises(ValueError):AttemptBudget(start,Clock())


if __name__=='__main__':unittest.main()
