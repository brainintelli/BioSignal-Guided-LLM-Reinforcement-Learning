"""Oddball v2 session controller:
warmup (G1) -> zero-output rehearsal -> 300/450/600 mV screening ladder (60 s rest between levels)
-> independent confirmation block (no escalation on fail) -> 12 main blocks -> recovery -> cleanup.

All device I/O sits behind `rig` so the full state machine is deterministically testable offline.
"""
import json
import os
import traceback

from . import analysis_v2 as analysis
from . import design_v2 as design
from .session import MAX_LATE_S, SETUP_MARGIN_S, START_LIMIT_S, Stop, TEARDOWN_S, WARMUP_HALF_S, write_once


class SessionV2:
    def __init__(self, rig, out, plan, start, session_id, bad_channels, clock, say=print):
        self.rig = rig
        self.out = out
        self.plan = design.check_plan(plan)
        self.start = start
        self.session_id = session_id
        self.bad_channels = list(bad_channels)
        self.clock = clock
        self.say = say
        self.stop_file = out / 'STOP'
        self.dose = 0
        self.record_open = False
        self.status = dict(schema=design.SCHEMA, blocks=[], started_wall=clock.wall())

    def age(self):
        return self.clock.wall() - self.start

    def note(self, text):
        self.say('[%5.1f min] %s' % (self.age() / 60.0, text), flush=True)

    def reserve(self, event):
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

    def run_block(self, block, bits):
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
        for k, (trials, handle) in enumerate(zip(spans, handles)):
            due = origin + k * period
            self.clock.sleep(max(0.0, due - self.clock.mono()))
            late = self.clock.mono() - due
            if late > MAX_LATE_S:
                raise Stop('host fell %.2f s behind chunk schedule in %s' % (late, name))
            event = dict(
                id=design.event_id(block, trials[0]),
                block=name,
                first=trials[0],
                trials=len(trials),
                real=design.real_pulses(block, trials),
                seconds=design.chunk_seconds(trials),
            )
            self.rig.send(handle, event, self.reserve)
        end = origin + (len(spans) - 1) * period + len(spans[-1]) * design.TRIAL_FRAMES / design.RATE
        self.clock.sleep(max(0.0, end - self.clock.mono()))
        self.rig.units(False)
        self.clock.sleep(design.POST_SECONDS)
        path = self.rig.close()
        self.record_open = False
        qc = self.rig.evaluate_block(path, block, self.session_id)
        write_once(self.out / (name + '-qc.json'), qc)
        self.status['blocks'].append(dict(name=name, stage=block['stage'], passed=qc['passed']))
        if not qc['passed']:
            raise Stop('closed-file check failed in ' + name)
        return qc

    def ladder_and_confirm(self):
        """Run 300 -> 450 -> 600 mV screening ladder; first qualifying amplitude enters independent confirm."""
        chosen_amp = None
        chosen_bits = None
        for idx, amp in enumerate(design.AMPLITUDES_MV):
            bits = design.pulse_bits(self.rig.lsb_mv, amp)
            blk = next(b for b in self.plan['blocks'] if b['stage'] == 'screen' and b['amplitude_mv'] == amp)
            qc = self.run_block(blk, bits)
            gate = analysis.screen_gate_v2(qc['rows'], self.plan['seed'] + idx)
            write_once(self.out / f'screen-{amp}.json', dict(amplitude_mv=amp, bits=bits, gate=gate))
            self.note('screen %d mV: passed sites %s' % (amp, gate['passed_sites']))
            if gate['qualified']:
                chosen_amp, chosen_bits = amp, bits
                break
            if idx + 1 < len(design.AMPLITUDES_MV):
                self.clock.sleep(design.SCREEN_REST_SECONDS)
        if chosen_bits is None:
            raise Stop('no amplitude in 300/450/600 mV passed screening gate (G2 pre-check)')

        self.clock.sleep(design.SCREEN_REST_SECONDS)
        confirm_blk = next(b for b in self.plan['blocks'] if b['stage'] == 'confirm')
        c_qc = self.run_block(confirm_blk, chosen_bits)
        c_gate = analysis.confirm_gate_v2(c_qc['rows'], self.plan['seed'] + 99)
        write_once(self.out / 'confirm.json', dict(amplitude_mv=chosen_amp, bits=chosen_bits, gate=c_gate))
        self.note('confirm %d mV: passed sites %s (qualified=%s)' % (chosen_amp, c_gate['passed_sites'], c_gate['qualified']))
        if not c_gate['qualified']:
            raise Stop(f'independent confirmation failed at {chosen_amp} mV; no further escalation allowed')
        write_once(self.out / 'decision.json', dict(
            amplitude_mv=chosen_amp,
            bits=chosen_bits,
            actual_mv=chosen_bits * self.rig.lsb_mv,
            passed_sites=c_gate['passed_sites'],
        ))
        return chosen_bits

    def settle(self):
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
        recover = False
        try:
            write_once(self.out / 'route.json', self.rig.prepare())
            # Warmup & G1 qualification
            self.rig.open('warmup')
            self.record_open = True
            self.clock.sleep(WARMUP_HALF_S)
            self.rig.units(True)
            self.clock.sleep(WARMUP_HALF_S)
            self.rig.units(False)
            self.clock.sleep(4.0)
            w_path = self.rig.close()
            self.record_open = False
            warm = self.rig.check_warmup(w_path)
            write_once(self.out / 'warmup.json', warm)
            if not warm.get('passed'):
                raise Stop('G1 warmup check failed: ' + warm.get('reason', 'unknown'))

            rehearsal = next(b for b in self.plan['blocks'] if b['stage'] == 'rehearsal')
            self.run_block(rehearsal, 0)
            bits = self.ladder_and_confirm()
            for blk in (b for b in self.plan['blocks'] if b['stage'] == 'main'):
                self.run_block(blk, bits)
            self.status['outcome'] = 'complete'
            recover = True
        except Stop as stop:
            self.status.update(outcome='stopped', stop_reason=str(stop))
            recover = True
        except Exception:
            self.status.update(outcome='error', error=traceback.format_exc())
            recover = True
        finally:
            if not self.rig.faulted:
                self.status['settle_errors'] = self.settle()
                if recover and self.dose and not self.status['settle_errors']:
                    rem = min(design.RECOVERY_SECONDS, design.HARD_STOP_S - TEARDOWN_S - 60 - self.age())
                    if rem >= 60.0:
                        self.rig.open('recovery')
                        self.record_open = True
                        self.clock.sleep(rem)
                        self.rig.close()
                        self.record_open = False
                        self.status['recovery'] = 'recorded %.0f s' % rem
            self.status['dose'] = self.dose
            self.status['finished_wall'] = self.clock.wall()
            write_once(self.out / 'end.json', self.status)
        return self.status
