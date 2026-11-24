"""One-shot reservation for exactly one image-parameter sequence."""
import json
import math
import os
from pathlib import Path


class TrialGuard:
    SEQUENCE_SECONDS = .8254
    POST_SECONDS = 600
    CLOSEOUT_SECONDS = 120
    REPLY_MARGIN_SECONDS = 20

    def __init__(self, root, *, started_wall, wall_now, monotonic_now):
        if not all(math.isfinite(v) for v in (started_wall, wall_now, monotonic_now)):
            raise ValueError('Nonfinite clock')
        if started_wall > wall_now:
            raise ValueError('Future session start')
        self.end_wall = started_wall + 7200
        self.end_mono = monotonic_now + self.end_wall - wall_now
        self.claim = Path(root)/'stimulus_send_claim.json'
        if self.remaining(wall_now=wall_now, monotonic_now=monotonic_now) <= 0:
            raise RuntimeError('Session expired')

    def remaining(self, *, wall_now, monotonic_now):
        return min(self.end_wall-wall_now, self.end_mono-monotonic_now)

    def reserve(self, *, wall_now, monotonic_now, qualified):
        if qualified is not True:
            raise RuntimeError('Post-connection baseline not qualified')
        required = self.SEQUENCE_SECONDS+self.POST_SECONDS+self.CLOSEOUT_SECONDS+self.REPLY_MARGIN_SECONDS
        if self.remaining(wall_now=wall_now, monotonic_now=monotonic_now) < required:
            raise RuntimeError('Insufficient full post-recording budget')
        with self.claim.open('x',encoding='utf-8') as f:
            json.dump({'reserved_pulses':15,'max_sequence_sends':1,'attempt_wall':wall_now,
                       'deadline_wall':self.end_wall,'no_automatic_retry':True},f,indent=2)
            f.flush();os.fsync(f.fileno())
