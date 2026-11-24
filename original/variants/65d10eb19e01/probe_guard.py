"""Fail-closed one-shot reservation and nonextending two-hour time budget."""
import json
import math
import os
from pathlib import Path


class ProbeGuard:
    MAX_SECONDS = 7200
    SEQUENCE_SECONDS = 40.0054
    SEND_REPLY_MARGIN_SECONDS = 20
    RECOVERY_SECONDS = 600
    CLOSEOUT_SECONDS = 120

    def __init__(self, root, *, started_wall, wall_now, monotonic_now):
        if not all(math.isfinite(v) for v in (started_wall, wall_now, monotonic_now)):
            raise ValueError('Nonfinite timestamp')
        if started_wall > wall_now:
            raise ValueError('Session has not started')
        self.deadline_wall = started_wall + self.MAX_SECONDS
        self.deadline_mono = monotonic_now + self.deadline_wall - wall_now
        self.claim = Path(root) / 'first_probe_send_claim.json'
        if self.remaining(wall_now=wall_now, monotonic_now=monotonic_now) <= 0:
            raise RuntimeError('Session expired')

    def remaining(self, *, wall_now, monotonic_now):
        return min(self.deadline_wall - wall_now, self.deadline_mono - monotonic_now)

    def reserve_send(self, *, wall_now, monotonic_now):
        remaining = self.remaining(wall_now=wall_now, monotonic_now=monotonic_now)
        if remaining < (self.SEQUENCE_SECONDS + self.SEND_REPLY_MARGIN_SECONDS +
                        self.RECOVERY_SECONDS + self.CLOSEOUT_SECONDS):
            raise RuntimeError('Insufficient time for sequence, recovery and closeout')
        record = {'reserved_pulses': 5, 'attempt_wall_time': wall_now,
                  'deadline_wall': self.deadline_wall,
                  'state': 'send_attempt_reserved_no_automatic_retry',
                  'note': 'Reservation is consumed even if the API reply is lost.'}
        # Create and durably persist before sending; never remove on failure.
        with self.claim.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
