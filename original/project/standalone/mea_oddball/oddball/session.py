"""Session sequencing, budgets and stop rules. All hardware sits behind `rig`.

A rig provides: prepare() -> route info; lsb_mv; site_channels; faulted;
open(name); build(block, bits) -> one handle per chunk; units(on);
send(handle, event, gate); close() -> path; read(path) -> recording;
verify(recording); cleanup() -> list of errors.
"""
import hashlib
import json
import os
import traceback

import numpy as np

from . import analysis, design

SETUP_MARGIN_S = 15.       # opening, building and closing one block file
MAX_LATE_S = .8            # a chunk this late is not sent
TEARDOWN_S = 120.
WARMUP_HALF_S = 30.
START_LIMIT_S = 35 * 60    # later than this, two complete rounds no longer fit


class Stop(Exception):
    """A registered reason to end stimulation. The recovery recording still runs."""


def write_once(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, allow_nan=False, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Session:
    def __init__(self, rig, out, plan, start, session_id, bad_channels, clock, say=print):
        self.rig, self.out, self.plan = rig, out, design.check_plan(plan)
        self.start, self.session_id, self.bad_channels = start, session_id, list(bad_channels)
        self.clock, self.say = clock, say
        self.stop_file = out / 'STOP'
        self.dose = 0
        self.record_open = False
        self.last_frame = None
        self.roi = self.channel_hz = self.reference_hz = None
        self.status = dict(blocks=[], started_wall=clock.wall())

    def age(self):
        return self.clock.wall() - self.start

    def note(self, text):
        self.say('[%5.1f min] %s' % (self.age() / 60, text), flush=True)

    # ---- budgets -----------------------------------------------------------------

    def reserve(self, event):
        """Last check before a chunk leaves the host; the dose is booked before it is sent."""
        real = event['real']
        if self.stop_file.exists():
            raise Stop('operator STOP file present')
        if real and self.age() + event['seconds'] > design.STIM_CUTOFF_S:
            raise Stop('stimulation cutoff reached')
        if self.dose + real > design.MAX_REAL_PULSES:
            raise Stop('session pulse budget reached')
        self.dose += real
        with (self.out / 'dose.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(event, total=self.dose, age_s=self.age())) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def admit(self, block):
        need = design.PRE_SECONDS + design.stimulation_seconds(block) + SETUP_MARGIN_S
        if self.stop_file.exists():
            raise Stop('operator STOP file present')
        if self.age() + need > design.STIM_CUTOFF_S or (block['stage'] == 'main' and self.age() > design.NEW_BLOCK_CUTOFF_S):
            raise Stop('not enough time left for block ' + block['name'])
        if self.dose + design.real_pulses(block) > design.MAX_REAL_PULSES:
            raise Stop('session pulse budget does not cover block ' + block['name'])

    # ---- files -------------------------------------------------------------------

    def opened(self, path):
        rec = self.rig.read(path)
        try:
            self.rig.verify(rec)
        except BaseException:
            rec.close()
            raise
        return rec

    def advance(self, rec):
        if self.last_frame is not None and rec.first_frame <= self.last_frame:
            raise ValueError('Device frame clock regressed between files')
        self.last_frame = rec.last_frame

    def warmup(self):
        """Quiet reference with the three units connected at neutral for the second half."""
        self.rig.open('warmup')
        self.record_open = True
        began = self.clock.mono()
        self.clock.sleep(WARMUP_HALF_S)
        self.rig.units(True)
        connected = self.clock.mono() - began
        self.clock.sleep(WARMUP_HALF_S)
        self.rig.units(False)
        self.clock.sleep(4)
        path = self.rig.close()
        self.record_open = False
        with self.opened(path) as rec:
            self.advance(rec)
            if rec.events or rec.duration < 2 * WARMUP_HALF_S:
                raise ValueError('Quiet reference is incomplete or contains an event')
            sites = [self.rig.site_channels[s] for s in design.SITES]
            for offset in (3, 12, 21):
                raw = rec.sample(sites, connected + offset, 1)
                if np.any((raw == 0) | (raw == 1023)):
                    raise ValueError('A connected site reached an ADC rail with zero output')
            chosen = analysis.select_roi(rec, self.rig.site_channels, self.bad_channels)
            counts = np.bincount(rec.spike_channels[rec.spike_times >= 4.], minlength=1024)
            self.channel_hz = {c: float(counts[c] / chosen['reference_seconds']) for c in chosen['roi']}
            self.set_roi(chosen['roi'])
            write_once(self.out / 'warmup.json', dict(chosen, source=rec.source, metadata=rec.metadata,
                                                    connected_at_s=connected, zero_output_site_rails=False))
        self.note('warmup: %d readout channels, reference %.1f spikes/s' % (len(self.roi), self.reference_hz))

    def set_roi(self, roi):
        self.roi = list(roi)
        self.reference_hz = float(sum(self.channel_hz[c] for c in self.roi))

    def block(self, block, bits):
        name = block['name']
        self.admit(block)
        self.rig.open(name)
        self.record_open = True
        handles = self.rig.build(block, bits)
        self.rig.units(True)
        self.clock.sleep(design.PRE_SECONDS)
        spans = design.chunks(block)
        period = design.CHUNK * design.TRIAL_FRAMES / design.RATE
        origin = self.clock.mono()
        sent = []
        for k, (trials, handle) in enumerate(zip(spans, handles)):
            due = origin + k * period
            self.clock.sleep(max(0., due - self.clock.mono()))
            late = self.clock.mono() - due
            if late > MAX_LATE_S:
                raise Stop('host fell %.2f s behind the chunk schedule in %s' % (late, name))
            event = dict(id=design.event_id(block, trials[0]), block=name, first=trials[0], trials=len(trials),
                         real=design.real_pulses(block, trials), seconds=design.chunk_seconds(trials))
            self.rig.send(handle, event, self.reserve)
            sent.append(dict(event, late_s=late, wall=self.clock.wall()))
        end = origin + (len(spans) - 1) * period + len(spans[-1]) * design.TRIAL_FRAMES / design.RATE
        self.clock.sleep(max(0., end - self.clock.mono()))
        self.rig.units(False)
        self.clock.sleep(design.POST_SECONDS)
        path = self.rig.close()
        self.record_open = False
        write_once(self.out / (name + '-sent.json'), dict(block=name, bits=bits, chunks=sent, path=str(path)))
        with self.opened(path) as rec:
            self.advance(rec)
            qc = analysis.block_qc(rec, block, self.session_id, self.rig.site_channels, self.roi, self.reference_hz)
        write_once(self.out / (name + '-qc.json'), qc)
        self.status['blocks'].append(dict(name=name, passed=qc['passed']))
        self.note('%s: %s, dose %d' % (name, 'passed' if qc['passed'] else 'FAILED ' + '; '.join(qc['errors'][:3]), self.dose))
        if not qc['passed']:
            raise Stop('closed-file check failed in ' + name)
        return dict(block=block, qc=qc, path=path)

    # ---- stages ------------------------------------------------------------------

    def screen(self):
        """Screen at 300 mV; escalate once to 400 mV only if fewer than two sites stand out."""
        warm_roi = list(self.roi)
        for amplitude in design.AMPLITUDES_MV:
            bits = design.pulse_bits(self.rig.lsb_mv, amplitude)
            self.set_roi(warm_roi)
            done = [self.block(b, bits) for b in self.plan['blocks']
                    if b['stage'] == 'screen' and b['amplitude_mv'] == amplitude]
            rails, real = [], []
            for item in done:
                with self.opened(item['path']) as rec:
                    rails.append(analysis.roi_rails(rec, [t['frame'] for t in item['qc']['trials']], warm_roi))
                real += [t['real'] for t in item['qc']['trials']]
            roi, dropped = analysis.prune_roi(warm_roi, np.concatenate(rails), real)
            rows = []
            if roi:
                for item in done:
                    with self.opened(item['path']) as rec:
                        values = analysis.responses(rec, [t['frame'] for t in item['qc']['trials']], roi)
                    rows += [dict(t, **v) for t, v in zip(item['qc']['trials'], values)]
            gate = analysis.screen_gate(rows, self.plan['seed']) if rows else dict(qualified=False, passed_sites=[])
            last = amplitude == design.AMPLITUDES_MV[-1]
            usable = len(roi) >= analysis.MIN_CHANNELS
            result = dict(amplitude_mv=amplitude, bits=bits, actual_mv=bits * self.rig.lsb_mv, roi=roi,
                          saturated_channels=dropped, gate=gate, proceed=bool(usable and (gate['qualified'] or last)))
            write_once(self.out / ('screen-%d.json' % amplitude), result)
            self.note('screen %d mV: sites told apart from sham %s, %d readout channels'
                      % (amplitude, gate['passed_sites'], len(roi)))
            if not usable:
                raise Stop('too few readout channels free of saturation at %d mV' % amplitude)
            if result['proceed']:
                self.set_roi(roi)
                write_once(self.out / 'decision.json', dict(
                    amplitude_mv=amplitude, bits=bits, actual_mv=result['actual_mv'], roi=roi,
                    reference_hz=self.reference_hz, qualified=gate['qualified'], passed_sites=gate['passed_sites'],
                    rule='300 mV if at least two sites pass, otherwise one escalation to 400 mV and proceed'))
                return bits
        raise AssertionError('unreachable')

    def recovery(self):
        seconds = min(design.RECOVERY_SECONDS, design.HARD_STOP_S - TEARDOWN_S - 60 - self.age())
        if seconds < 60:
            self.status['recovery'] = 'skipped: no time before the hard stop'
            return
        self.rig.open('recovery')
        self.record_open = True
        self.clock.sleep(seconds)
        path = self.rig.close()
        self.record_open = False
        with self.opened(path) as rec:
            self.advance(rec)
            spikes = rec.spike_times[np.isin(rec.spike_channels, self.roi)]
            rates = [float(np.count_nonzero((spikes >= a) & (spikes < a + 30)) / 30)
                     for a in range(0, int(rec.duration) - 29, 30)]
            write_once(self.out / 'recovery.json', dict(
                source=rec.source, seconds=float(rec.duration), events=len(rec.events), reference_hz=self.reference_hz,
                rate_30s_hz=rates, ratio_30s=[r / self.reference_hz for r in rates]))
        self.status['recovery'] = 'recorded %.0f s' % seconds
        self.note('recovery recorded')

    def settle(self):
        """Leave the device quiet with no open file after any exit from the stimulation stages."""
        errors = []
        for label, action in (('units off', lambda: self.rig.units(False)),
                              ('close file', lambda: self.record_open and self.rig.close())):
            try:
                action()
            except Exception as exc:
                errors.append(label + ': ' + repr(exc))
        self.record_open = False
        return errors

    def run(self):
        status = self.status
        recover = False
        try:
            write_once(self.out / 'route.json', self.rig.prepare())
            self.warmup()
            self.block(next(b for b in self.plan['blocks'] if b['stage'] == 'rehearsal'), 0)
            bits = self.screen()
            for block in (b for b in self.plan['blocks'] if b['stage'] == 'main'):
                self.block(block, bits)
            status['outcome'] = 'complete'
            recover = True
        except Stop as stop:
            status.update(outcome='stopped', stop_reason=str(stop))
            self.note('stimulation stopped: %s' % stop)
            recover = True
        except Exception:
            status.update(outcome='error', error=traceback.format_exc())
            recover = True
        except BaseException:
            status.update(outcome='interrupted', error=traceback.format_exc())
        finally:
            try:
                if not self.rig.faulted:
                    status['settle_errors'] = self.settle()
                    if recover and self.dose and not status['settle_errors']:
                        try:
                            self.recovery()
                        except Exception:
                            status['recovery'] = 'failed: ' + traceback.format_exc()
                            self.settle()
            finally:
                status['cleanup'] = self.rig.cleanup()
                status.update(dose=self.dose, finished_wall=self.clock.wall(), age_s=self.age())
                write_once(self.out / 'end.json', status)
        return status

    def manifest(self):
        """Hash every raw file. Runs after the device is released; it can take minutes."""
        files = sorted(self.out.glob('*/*.raw.h5'))
        write_once(self.out / 'manifest.json', {p.relative_to(self.out).as_posix(): dict(
            bytes=p.stat().st_size, sha256=file_sha256(p)) for p in files})
