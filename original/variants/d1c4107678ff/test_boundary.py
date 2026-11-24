import unittest
from batch_boundary import bootstrap,validate_tail

class BoundaryTests(unittest.TestCase):
    def test_terminal_skips_model_calls_even_with_reset_observation(self):
        def forbidden(_):raise AssertionError('terminal/reset observation used')
        self.assertEqual(bootstrap([{'terminal':True}],{'dead':False},forbidden,forbidden),(0.,None))
    def test_nonterminal_keeps_value(self):
        self.assertEqual(bootstrap([{'terminal':False}],{'n':2},lambda x:('input',x),lambda p:3.25),(3.25,('input',{'n':2})))
    def test_tail_not_early_death_or_timeout(self):
        validate_tail([{'terminal':False},{'terminal':True}],2)
        for rows,n in [([{'terminal':True},{'terminal':False}],2),([{'terminal':False}]*2,2),([{'terminal':True}],2)]:
            with self.assertRaises(AssertionError):validate_tail(rows,n)
if __name__=='__main__':unittest.main()
