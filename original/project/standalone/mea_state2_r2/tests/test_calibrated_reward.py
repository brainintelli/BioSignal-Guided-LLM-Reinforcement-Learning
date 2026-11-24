import unittest

from mea_state2_r2.reward_mix import calibrated_reward, recording_mix, validate_mix

SHA='2fc7e0211eb3267b2ce31f0a2e494e658c1696efbae078f496291851a69028c7'


class CalibratedRewardTests(unittest.TestCase):
    def test_bound_survives_duration_and_both_signs(self):
        mix=recording_mix(SHA)
        for native in (-1.,-1e-8,0.,1e-8,.02):
            for ticks in (1,4,16):
                for label in (-1,1,None):
                    scaled,weighted=calibrated_reward(native,label,ticks,.99985,mix)
                    self.assertLessEqual(abs(scaled),.5*abs(native))
                    self.assertLessEqual(abs(weighted),.05*abs(native))
                    self.assertAlmostEqual(weighted,.1*scaled)
                    if native and label is not None:
                        self.assertGreater(label*scaled,0)
                        self.assertGreater(native*(native+weighted),0)
                    else:self.assertEqual((scaled,weighted),(0.,0.))

    def test_reference_is_frozen_and_weight_is_independent(self):
        mix=recording_mix(SHA)
        scaled,weighted=calibrated_reward(.1,1,1,.99985,mix)
        self.assertAlmostEqual(scaled,.5*mix['reference_per_tick'])
        other=validate_mix(dict(mix,r2_weight=.2))
        self.assertEqual(calibrated_reward(.1,1,1,.99985,other),(scaled,weighted*2))
        with self.assertRaises(ValueError):recording_mix('wrong')
        with self.assertRaises(ValueError):validate_mix(dict(mix,amplitude_fraction=1))


if __name__=='__main__':unittest.main()
