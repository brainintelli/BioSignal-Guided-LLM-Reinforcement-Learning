"""One finite 15-pulse train per explicitly bound point; no device access on import.

The caller owns the 20-point plan, dose reservation, live authorization, biological
deadline and pre/post recording. This module only builds and sends one train.
It intentionally reuses the frozen v2 SDK audit, ownership and cleanup behavior.
"""
from pathlib import Path
import re
import sys
import uuid

V2_ROOT = Path(__file__).resolve().parent.parent / 'mea_stimulation_v2'
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from mea_stim_v2.live import MaxLabBackend, ManagedSequence, mx, _operation
from mea_stim_v2.protocol import pulse_codes


ONSET_SAMPLES = tuple(100 + group * 4000 + pulse * 200
                      for group in range(5) for pulse in range(3))
END_SAMPLES = 16508


class MultipointBackend(MaxLabBackend):
    """Each instance permits at most one attempted train at its bound electrode."""

    def _train_events(self, events):
        if not isinstance(events, (list, tuple)) or len(events) != 15:
            raise ValueError('Exactly 15 ordered events are required')
        normalized = []
        ids = set()
        point = None
        required = {'id', 'arm', 'pair', 'phase', 'point', 'electrode', 'group', 'pulse'}
        for index, supplied in enumerate(events):
            if not isinstance(supplied, dict) or not required <= supplied.keys() or set(supplied) - required - {'session'}:
                raise ValueError('Unexpected or missing train event properties')
            event = {**supplied, 'session': self.config.session_id}
            if supplied.get('session', self.config.session_id) != self.config.session_id:
                raise ValueError('Event session differs from the live binding')
            if any(type(event[key]) is not int for key in ('id', 'pair', 'point', 'electrode', 'group', 'pulse')):
                raise ValueError('Train identity and positions must be integers')
            if not 1 <= event['id'] < 2**20 or event['id'] in ids or event['id'] in self._events_attempted:
                raise ValueError('Invalid, duplicate or previously attempted event ID')
            if point is None:
                point = event['point']
            if not 1 <= point <= 20 or event['point'] != point or event['pair'] != point:
                raise ValueError('All events must belong to one point/pair in 1..20')
            if event['electrode'] != self.config.stim_electrode:
                raise ValueError('Train electrode differs from the bound recording route')
            if event['arm'] != 'true' or event['phase'] != 'multipoint':
                raise ValueError('Only true multipoint events are supported')
            if event['group'] != index // 3 + 1 or event['pulse'] != index % 3 + 1:
                raise ValueError('Expected five groups of three pulses in 1-based order')
            for value in event.values():
                if type(value) is not int and not (isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', value)):
                    raise ValueError('Unsafe event property value')
            encoded = ' '.join(f'{key} {value}' for key, value in sorted(event.items()))
            if len(encoded.encode('ascii')) > 240:
                raise ValueError('Train event properties exceed the metadata budget')
            ids.add(event['id'])
            normalized.append((event, encoded))
        return normalized

    @_operation
    def train(self, events15, lsb_mv):
        """Send exactly one finite train; the controller reserves all 15 first.

        before_sequence_send receives the first event, once, immediately before
        the single send. Its rejection prevents the entire train from sending.
        Once sent, this finite 0.8254 s train runs without host-side interleaving.
        """
        if getattr(self, '_train_attempted', False):
            raise RuntimeError('This point already attempted its single train')
        events = self._train_events(events15)
        codes = pulse_codes('true', lsb_mv)
        if not self.file_open or not self.recording or not self.connected:
            raise RuntimeError('Train requires active recording and the selected connected unit')
        self.assert_ready()
        with self._sdk():
            actual_lsb = float(mx.query_DAC_lsb_mV())
            pulse_codes('true', actual_lsb)
            if actual_lsb != lsb_mv:
                self._faulted = True
                raise RuntimeError('DAC step changed since its admitted value')
            self._train_attempted = True
            self._retire_sequences()
            token = 'mp_seq_' + uuid.uuid4().hex
            self.sequences.append(token)
            self._events_attempted.update(event['id'] for event, _ in events)
            # Ownership precedes constructor and all API traffic; lost ACKs
            # cannot lose the resource or permit automatic re-sending.
            sequence = ManagedSequence(token=token, initial_delay=100, persistent=True)
            if sequence.token != token:
                raise RuntimeError('Server returned a different sequence token')
            for index, (event, encoded) in enumerate(events):
                sequence.append(mx.Event(0, 1, event['id'], encoded))
                for kind, value in codes:
                    sequence.append(mx.DAC(0, value) if kind == 'dac' else mx.DelaySamples(value))
                if index < 14:
                    # Each biphasic pulse consumes 8 samples. The inter-pulse
                    # gap is measured from its end to the next prescribed onset.
                    sequence.append(mx.DelaySamples(ONSET_SAMPLES[index+1] - ONSET_SAMPLES[index] - 8))
            self._pending_event = dict(events[0][0])
            self._pending_pulse = True
            try:
                sequence.send()
            finally:
                self._pending_event = None
                self._pending_pulse = False
        self.clock.sleep(.826)
        return {'send_mono': self.sent_monotonic_actual,
                'event': dict(self.sent_event), 'events': [dict(event) for event, _ in events],
                'lsb_mv': actual_lsb, 'amplitude_mv': int(300 / actual_lsb) * actual_lsb,
                'onset_samples': list(ONSET_SAMPLES), 'duration_samples': END_SAMPLES,
                'duration_s': END_SAMPLES / 20000}
