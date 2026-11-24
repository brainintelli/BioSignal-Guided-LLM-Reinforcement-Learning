from pathlib import Path
import tempfile
import unittest
from probe_guard import ProbeGuard


class GuardTests(unittest.TestCase):
    def make(self, root, wall=1000, mono=30):
        return ProbeGuard(Path(root), started_wall=900, wall_now=wall, monotonic_now=mono)

    def test_one_send_including_a_new_process(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.make(d)
            g.reserve_send(wall_now=1600, monotonic_now=630)
            self.assertEqual(g.claim.read_text().count('reserved_pulses'), 1)
            with self.assertRaises(FileExistsError):
                g.reserve_send(wall_now=1601, monotonic_now=631)
            with self.assertRaises(FileExistsError):
                self.make(d, wall=1700).reserve_send(wall_now=1700, monotonic_now=30)

    def test_lost_response_keeps_claim(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.make(d)
            def uncertain_send():
                g.reserve_send(wall_now=1600, monotonic_now=630)
                raise TimeoutError('after device may have acted')
            with self.assertRaises(TimeoutError):
                uncertain_send()
            self.assertTrue(g.claim.exists())

    def test_full_recovery_closeout_reserved(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.make(d)
            with self.assertRaises(RuntimeError):
                g.reserve_send(wall_now=7320, monotonic_now=6350)
            self.assertFalse(g.claim.exists())
            g.reserve_send(wall_now=7319, monotonic_now=6349)

    def test_clock_rollback_cannot_extend_session(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.make(d)
            with self.assertRaises(RuntimeError):
                g.reserve_send(wall_now=1600, monotonic_now=6370)

    def test_expired_or_future_start_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self.make(d, wall=800)
            with self.assertRaises(RuntimeError):
                self.make(d, wall=8101)

if __name__ == '__main__':
    unittest.main()
