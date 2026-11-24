"""Behavior fixtures derive from the protocol, not compiler internals."""
import dataclasses
import unittest

try:
    from mea_binary_r2.encoding import Encoder, Intent, compile_sequence
except ImportError:
    Encoder = Intent = compile_sequence = None


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(Encoder, 'stimulus encoder has not been implemented')

    def test_hysteresis_and_recovery(self):
        encoder = Encoder()
        self.assertEqual(encoder.decide({'H':100,'F':100,'W':85}).pulses, 0)
        for water in (84,85,89):
            choice = encoder.decide({'H':100,'F':100,'W':water})
            self.assertEqual((choice.site, choice.pulses), ('W',1))
        self.assertEqual(encoder.decide({'H':100,'F':100,'W':90}).pulses, 0)

    def test_worst_reserve_ties_and_severity(self):
        cases = [({'H':70,'F':60,'W':65},'F',1),
                 ({'H':50,'F':50,'W':50},'H',3),
                 ({'H':100,'F':60,'W':60},'W',1),
                 ({'H':100,'F':100,'W':50.01},'W',1),
                 ({'H':100,'F':100,'W':0},'W',3)]
        for reserves,site,pulses in cases:
            with self.subTest(reserves=reserves):
                got=Encoder().decide(reserves)
                self.assertEqual((got.site,got.pulses),(site,pulses))

    def test_invalid_observation_does_not_mutate_latches(self):
        encoder=Encoder()
        encoder.decide({'H':100,'F':100,'W':80})
        for reserves in ({'H':80,'F':100}, {'H':True,'F':100,'W':100},
                         {'H':float('nan'),'F':100,'W':100},
                         {'H':100,'F':float('inf'),'W':100},
                         {'H':100,'F':100,'W':-1},
                         {'H':100,'F':100,'W':101},
                         {'H':100,'F':100,'W':100,'extra':1}):
            with self.subTest(reserves=reserves), self.assertRaises(ValueError):
                encoder.decide(reserves)
        self.assertEqual(encoder.decide({'H':100,'F':100,'W':87}).site,'W')

    def test_terminal_is_ineligible_and_does_not_stimulate(self):
        terminal=Encoder().decide({'H':0,'F':0,'W':0},terminal=True)
        self.assertEqual((terminal.site,terminal.pulses,terminal.eligible),(None,0,False))
        self.assertEqual(compile_sequence(terminal,2.9),())
        with self.assertRaises(ValueError):
            Encoder().decide({'H':100,'F':100,'W':100},terminal=1)

    def test_intent_is_frozen_and_consistent(self):
        intent=Encoder().decide({'H':100,'F':100,'W':80})
        with self.assertRaises(dataclasses.FrozenInstanceError):intent.pulses=3
        for args in ((None,1,True,()),('W',0,True,('W',)),('W',2,True,('W',)),
                     ('X',1,True,('X',)),('W',1,False,('W',))):
            with self.subTest(args=args),self.assertRaises(ValueError):Intent(*args)

    def test_quantization_and_zero_return(self):
        intent=Encoder().decide({'H':100,'F':100,'W':80})
        # floor(300/2.9)=103 DAC steps; actual amplitude=298.7 mV.
        self.assertEqual(compile_sequence(intent,2.9),
                         (('dac',409),('delay',4),('dac',615),('delay',4),('dac',512)))
        self.assertEqual(compile_sequence(Encoder().decide({'H':100,'F':100,'W':100}),2.9),())

    def test_three_pulses_use_start_to_start_300ms(self):
        sequence=compile_sequence(Encoder().decide({'H':100,'F':100,'W':20}),3.)
        elapsed=0;starts=[]
        for command,value in sequence:
            if command=='delay':elapsed+=value
            elif value==412:starts.append(elapsed)
        self.assertEqual(starts,[0,6000,12000])
        self.assertEqual(elapsed,12008)
        self.assertEqual(sequence[-1],('dac',512))

    def test_unrepresentable_or_invalid_dac_lsb_is_rejected(self):
        intent=Encoder().decide({'H':80,'F':100,'W':100})
        for lsb in (0,-1,True,float('nan'),float('inf'),1000,.1):
            with self.subTest(lsb=lsb),self.assertRaises(ValueError):compile_sequence(intent,lsb)


if __name__=='__main__':unittest.main()
