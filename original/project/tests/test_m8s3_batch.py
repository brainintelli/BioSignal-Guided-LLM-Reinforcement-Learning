"""Canonical batching must not change a row's layout when peers change."""
import unittest
import copy


class BatchLayoutTests(unittest.TestCase):
    def batches(self, rows):
        from agent.m8s3_batch_layout import canonical_batches
        return list(canonical_batches(rows, boundaries=(4, 8, 12), batch_size=4, pad_id=0))

    def test_rows_keep_their_width_and_tokens_after_regrouping(self):
        rows = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
        result = self.batches(rows)
        self.assertEqual([b['indices'] for b in result], [[0, 2], [1]])
        self.assertEqual(result[0]['input_ids'], [[0, 1, 2, 3], [0, 0, 0, 9], [0, 0, 0, 9], [0, 0, 0, 9]])
        self.assertEqual(result[1]['attention_mask'], [[0, 0, 0, 1, 1, 1, 1, 1]] * 4)
        for i, row in enumerate(rows):
            alone = self.batches([row])[0]
            group = next(b for b in result if i in b['indices'])
            self.assertEqual(alone['input_ids'][0], group['input_ids'][group['indices'].index(i)])

    def test_batch_overflow_preserves_every_real_row_once(self):
        result = self.batches([[i] for i in range(1, 7)])
        self.assertEqual([b['indices'] for b in result], [[0, 1, 2, 3], [4, 5]])
        self.assertTrue(all(len(b['input_ids']) == 4 for b in result))

    def test_exact_boundary_never_removes_the_padding_mask(self):
        # An all-ones mask lets SDPA choose a different numerical kernel when
        # the same row is evaluated alone rather than with a padded peer.
        for row in ([1] * 4, [1] * 8, [1] * 11):
            alone = self.batches([row])[0]
            self.assertEqual(alone['attention_mask'][0][0], 0)
            self.assertGreater(len(alone['input_ids'][0]), len(row))
        from agent.m8s3_batch_layout import canonical_batches
        full = list(canonical_batches([[1] * 1536]))[0]
        self.assertEqual(sum(full['attention_mask'][0]), 1536)
        self.assertEqual(full['attention_mask'][0][0], 0)
        with self.assertRaises(ValueError):
            list(canonical_batches([[1] * 1537]))

    def test_invalid_input_is_rejected_without_truncation(self):
        from agent.m8s3_batch_layout import canonical_batches
        for rows in ([], [[]], [[1] * 13]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.batches(rows)
        for boundaries in ((8, 4), (4, 4), (), (0, 4)):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                list(canonical_batches([[1]], boundaries=boundaries))


class BatchRegistrationTests(unittest.TestCase):
    def test_registration_refuses_stale_rollouts_or_unregistered_layout(self):
        from agent.m8s3_batch_contract import fixed_config, validate_config
        config = fixed_config('/remote/project', '/remote')
        validate_config(config)
        for change in ({'fresh_rollouts': False}, {'reset_critic': False},
                       {'candidate_layout': 'batch_max_dynamic'}, {'microbatch': 2},
                       {'seeds': [883910, 883911, 883912, 883913]},
                       {'formal_training_status': 'running'}):
            mutated = copy.deepcopy(config)
            mutated.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(mutated)


if __name__ == '__main__':
    unittest.main()
