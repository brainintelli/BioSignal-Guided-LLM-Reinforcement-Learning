"""Context expansion preserves every token and the historical physical layouts."""
import importlib
import json
import os
from pathlib import Path
import unittest


class ContextTests(unittest.TestCase):
    def context(self):
        try:
            return importlib.import_module('agent.m9_context')
        except ModuleNotFoundError:
            self.fail('M9 context extension is missing')

    def test_previous_states_keep_identical_padded_batches(self):
        from agent.m8s3_batch_layout import canonical_batches
        module = self.context()
        old = module.context_layout(1536)
        new = module.context_layout(4096)
        for length in (1, 895, 896, 1279, 1280, 1536):
            ids = [7] * length
            self.assertEqual(list(canonical_batches([ids], old)),
                             list(canonical_batches([ids], new)))

    def test_overflow_state_and_full_continuation_headroom(self):
        from agent.m8s3_batch_layout import canonical_batches
        module = self.context()
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages
            def encode(self, text, **kwargs):
                return text
        for length in (1541, 2048, 3072, 4032):
            ids = list(range(length))
            actual = module.tokenize_observation(Tokenizer(), ids, 4096)
            self.assertEqual(actual, ids)
            batch = list(canonical_batches([actual], module.context_layout(4096)))[0]
            self.assertEqual(sum(batch['attention_mask'][0]), length)
            self.assertEqual(batch['attention_mask'][0][0], 0)
        with self.assertRaises(ValueError):
            module.tokenize_observation(Tokenizer(), list(range(1541)), 1536)
        with self.assertRaises(ValueError):
            module.tokenize_observation(Tokenizer(), list(range(4033)), 4096)
        for invalid in (0, 1541, 8192, True):
            with self.assertRaises(ValueError):
                module.context_layout(invalid)

    @unittest.skipUnless(os.environ.get('MEADOW_MODEL'), 'Requires local node tokenizer')
    def test_actual_failed_observation_is_losslessly_accepted(self):
        from transformers import AutoTokenizer
        from agent.m9_contract import encode
        module = self.context()
        fixture = json.loads((Path(__file__).parent / 'fixtures/m9_token_overflow.json').read_text(encoding='utf-8'))
        tokenizer = AutoTokenizer.from_pretrained(os.environ['MEADOW_MODEL'], local_files_only=True)
        messages = encode(fixture['observation'], 'compact')['messages']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(len(ids), 1541)
        with self.assertRaisesRegex(ValueError, '1541.*1536'):
            module.tokenize_observation(tokenizer, messages, 1536)
        self.assertEqual(module.tokenize_observation(tokenizer, messages, 4096), ids)


if __name__ == '__main__':
    unittest.main()
