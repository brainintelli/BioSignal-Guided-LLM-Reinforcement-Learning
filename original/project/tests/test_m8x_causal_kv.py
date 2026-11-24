"""A continuation must preserve every target token after the assistant boundary."""
import importlib
import importlib.util
import unittest

class ContinuationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_causal_kv'))
        return importlib.import_module('agent.m8x_causal_kv')

    def test_empty_thinking_stub_inserted_without_dropping_target_tokens(self):
        fn=self.module().normalize_target
        self.assertEqual(fn([1,2,3,7,8],[1,2,3,21,22,23],3,12),[1,2,3,7,8,21,22,23])

    def test_mismatched_prefix_or_expanded_overflow_rejected(self):
        fn=self.module().normalize_target
        for verb,target,boundary,limit in [([1,2,7,8],[1,99,21,22],2,10),([1,2,7,8],[1,2,21,22],2,5)]:
            with self.assertRaises(ValueError):fn(verb,target,boundary,limit)

if __name__=='__main__':unittest.main()
