"""Local Linux MaxLab adapter. Import and preview never connect to hardware."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from brain_stimulation import maxlab as mx
from brain_stimulation.runner import ManagedSequence,build_sequence
from brain_stimulation.plan import StimulationParameters
from brain_stimulation.transport import Api
from protocol import PROFILE,atomic_json
from quality import parse_mapping,inspect_h5


def compile_preview(lsb):
    if not math.isfinite(lsb) or lsb<=0 or not 1<=int(600/lsb)<=511:
        raise ValueError('Current DAC LSB cannot realize the bounded bipolar profile')
    params=StimulationParameters.from_params(PROFILE)
    class Capture:
        def __init__(self):self.commands=['system_delay_samples 100']
        def append(self,obj):self.commands.append(obj.set())
    seq=Capture();build_sequence(params,[600]*5,lsb,seq)
    ticks=0;onsets=[]
    for line in seq.commands:
        parts=line.split()
        if parts[0]=='system_delay_samples':ticks+=int(parts[1])
        elif parts[0]=='system_trigger_event':onsets.append(ticks*.05)
    bits=int(600/lsb)
    return dict(status='software_preview_not_measured_voltage',lsb_mv=lsb,
                codes=[512-bits,512+bits,512],nominal_phase_mv=bits*lsb,
                pulse_onset_ms=onsets,duration_s=ticks*.00005,commands=seq.commands)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class MaxLabBackend:
    def __init__(self,binding,output,*,request=None,clock=time):
        self.b=binding;self.out=Path(output);self.clock=clock
        self.request=request or self._network_request
        self.wait=clock.sleep;self.controller=None
        self.touched=False;self.file_attempted=False;self.record_attempted=False
        self.array_token=None;self.sequence_tokens=[];self.cleaning=False
        self.current_file=None;self.record_times={};self.logs=[];self.candidate=None
        self.route=Path(binding['route_cfg']).read_text()
        self.mapping=parse_mapping(self.route)
        self.last_growth=None;self.last_size=0;self.lease=None;self.growth_events=[]

    def _network_request(self,message,timeout):
        with socket.create_connection(('127.0.0.1',7215),timeout=min(2,timeout)) as s:
            return Api(s,timeout).send(message)

    @contextmanager
    def sdk_context(self):
        old=mx.comm.api_context;owner=self
        class Bound:
            def send(self,message,timeout=None):return owner._raw(message)
        @contextmanager
        def context(*a,**kw):yield Bound()
        mx.comm.api_context=context
        try:yield
        finally:mx.comm.api_context=old

    def _raw(self,message):
        if not self.cleaning and self.controller:self.controller.deadline.require(120)
        if message.startswith('mea_set_dac ') and message!='mea_set_dac 0 512 512':
            raise RuntimeError('Direct nonzero DAC forbidden')
        entry=dict(wall=self.clock.time(),mono=self.clock.monotonic(),command=message)
        self.logs.append(entry)
        # Record intent before a potentially ambiguous network action.
        self._log(entry)
        try:
            reply=self.request(message,2 if self.cleaning else 5)
            if str(reply).strip().lower().startswith('error'):raise RuntimeError(str(reply))
        except BaseException as exc:
            self._log(dict(error=repr(exc),command=message));raise
        self._log(dict(ack=reply,command=message,wall=self.clock.time()))
        return reply

    def _log(self,entry):
        try:
            with (self.out/'api_audit.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush()
                if entry.get('command','').startswith('sequence_send'):os.fsync(f.fileno())
        except OSError:
            if not self.cleaning:raise

    def preflight(self):
        if sys.platform!='linux':raise RuntimeError('Actual runs require the Linux acquisition host')
        # Host identity, not a remotely directed endpoint. No DNS-based substitution.
        interfaces=json.loads(subprocess.check_output(['ip','-j','addr'],text=True))
        addresses={a['local'] for i in interfaces for a in i.get('addr_info',[])}
        if self.b['acquisition_host'] not in addresses:raise RuntimeError('Wrong local acquisition host')
        import fcntl
        lock=Path.home()/'.local/state/mea_photo_full/device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        self.lease=lock.open('a');fcntl.flock(self.lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        servers=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if (p/'comm').read_text().strip()!='mxwserver':continue
                servers.append(p.name)
                for fd in (p/'fd').iterdir():
                    try:target=os.readlink(fd)
                    except FileNotFoundError:continue
                    if '.h5' in target:raise RuntimeError('MaxLab already has an open H5: '+target)
            except FileNotFoundError:continue
            # PermissionError intentionally propagates: unknown busy state is not idle.
        if len(servers)!=1:raise RuntimeError('Require exactly one inspectable mxwserver')
        if shutil.disk_usage(self.out).free<120*1024**3:raise RuntimeError('Need 120 GiB free for uncompressed raw budget')
        if str(self.out).encode('ascii',errors='ignore').decode()!=str(self.out) or any(c in str(self.out) for c in '\r\n\x00') or '###' in str(self.out):
            raise ValueError('Recording path must be ASCII with no protocol delimiters')
        if self._raw('wellplate_query_version').strip()!='0':raise RuntimeError('Ordinary MaxOne required')
        atomic_json(self.out/'source_manifest.json',{str(p.relative_to(Path(__file__).parent)):sha256(p)
                    for p in Path(__file__).parent.rglob('*.py')})
        (self.out/'route_input.cfg').write_text(self.route)

    def zero(self):
        # All independent shutdowns are attempted, even when one response is lost.
        errors=[]
        for unit in range(32):
            try:mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
            except Exception as exc:errors.append(f'unit {unit}: {exc!r}')
        try:mx.send(mx.DAC(0,512))
        except Exception as exc:errors.append('DAC: '+repr(exc))
        if errors:raise RuntimeError('; '.join(errors))

    def setup_identity(self):
        self.touched=True;self.zero();mx.clear_events();mx.offset()

    def open(self,name):
        if name not in ('identity','baseline','stimulus_post'):raise ValueError('Unexpected file stem')
        self.current_file=self.out/(name+'.raw.h5')
        if self.current_file.exists():raise FileExistsError(str(self.current_file))
        s=mx.Saving();s.open_directory(str(self.out));s.set_legacy_format(False)
        self.file_attempted=True;s.start_file(name)
        for key,value in {'protocol':'photo_full_v1','alpha':'1','r2':'record_only',
                          'training':'disabled','phase':name}.items():s.write_assay_input(key,value)
        s.group_delete_all();s.group_define(0,'routed',list(range(1024)))
        self.record_attempted=True;s.start_recording([0])
        self.record_times[name]={'start':self.clock.monotonic()}
        self.last_growth=self.clock.monotonic();self.last_size=0
        self.growth_events=[]

    def close(self):
        s=mx.Saving()
        if self.record_attempted:s.stop_recording();self.record_attempted=False
        if self.file_attempted:s.stop_file();self.file_attempted=False
        if self.current_file:
            self.record_times[self.current_file.name.split('.')[0]]['end']=self.clock.monotonic()
            atomic_json(self.out/'recording_times.json',self.record_times)

    def _inspect(self,name,seconds,phase,selected=None):
        heartbeat=(lambda:self.controller.deadline.require(120)) if self.controller and not self.cleaning else None
        result=inspect_h5(self.out/(name+'.raw.h5'),self.b['chip_id'],self.mapping,seconds,
                          phase=phase,selected=selected,heartbeat=heartbeat)
        atomic_json(self.out/(name+'_qc.json'),result)
        return result

    def identity(self):
        result=self._inspect('identity',60,'identity')
        if result['status']!='passed':raise RuntimeError('Fresh identity rejected: '+repr(result['errors']))
        self.candidate=result['candidate'];return self.candidate

    def configure(self,candidate):
        self.candidate=candidate;self.touched=True
        mx.initialize();self.zero();mx.send(mx.Core().enable_stimulation_power(True))
        self.array_token='full_'+uuid.uuid4().hex
        self.array=mx.Array(self.array_token,persistent=True);self.array.load_config_data(self.route)
        electrode=candidate['electrode'];ch=candidate['channel']
        # A loaded cfg may contain multiple amplifiers sharing the chosen unit.
        # Clear every inherited stimulation connection, including unmapped amps.
        for amplifier in range(1024):self.array.disconnect_amplifier_from_stimulation(amplifier)
        self.array.connect_electrode_to_stimulation(electrode)
        unit=int(self.array.query_stimulation_at_electrode(electrode).strip())
        if not 0<=unit<32:raise RuntimeError('Invalid stimulation unit')
        for amplifier in range(1024):
            result=self.array.query_stimulation_at_amplifier(amplifier).strip()
            expected=str(unit) if amplifier==ch else None
            if (expected is not None and result!=expected) or (expected is None and result not in ('','-1')):
                raise RuntimeError('Additional or unknown stimulation connection at amplifier '+str(amplifier))
        mx.activate([0]);self.array.download([0]);self.wait(5)
        config=self._raw('mea_array_command '+self.array_token+'\nmea_array_get_config')
        if parse_mapping(config)!=self.mapping:raise RuntimeError('Downloaded mapping changed')
        if (int(self.array.query_amplifier_at_electrode(electrode))!=ch or
            int(self.array.query_amplifier_at_stimulation(unit))!=ch):raise RuntimeError('Bidirectional route mismatch')
        (self.out/'downloaded.cfg').write_text(config)
        mx.offset();self.wait(10)
        mx.send(mx.DAC(0,512));mx.send(mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(0))
        self.unit=unit
        atomic_json(self.out/'selected_route.json',dict(**candidate,stimulation_unit=unit,route_sha256=hashlib.sha256(config.encode()).hexdigest()))

    def prepare(self):
        lsb=float(mx.query_DAC_lsb_mV());self.preview=compile_preview(lsb)
        atomic_json(self.out/'sequence_preview.json',self.preview)
        mx.clear_events()
        token='full_pulse_'+uuid.uuid4().hex;self.sequence_tokens.append(token)
        sequence=ManagedSequence(token=token,initial_delay=0,persistent=True)
        if sequence.token!=token:raise RuntimeError('Unexpected sequence token')
        sequence.append(mx.DelaySamples(100))
        build_sequence(StimulationParameters.from_params(PROFILE),[600]*5,lsb,sequence)
        self.pulse_token=token
        token='full_marker_'+uuid.uuid4().hex;self.sequence_tokens.append(token)
        marker=ManagedSequence(token=token,initial_delay=0,persistent=True)
        if marker.token!=token:raise RuntimeError('Unexpected marker token')
        marker.append(mx.DelaySamples(100));marker.append(mx.Event(0,1,7001,'baseline_no_nonzero_output'))
        self.marker_token=token

    def marker(self):
        if self.current_file.name!='baseline.raw.h5':raise RuntimeError('Marker outside baseline')
        reply=self._raw('sequence_send '+self.marker_token)
        if reply.strip().lower()!='ok':raise RuntimeError('Baseline marker acknowledgement unknown')

    def qualify(self):return self._inspect('baseline',600,'baseline',self.candidate)

    def send(self):
        c=self.controller
        if (not c or c.report.get('stage')!='single_stimulus_attempt'
            or c.report.get('stimulus_attempted') is not True
            or c.report.get('baseline_qc',{}).get('status')!='passed'):
            raise RuntimeError('Unqualified output attempt')
        self.verify_recording_ready()
        reply=self._raw('sequence_send '+self.pulse_token)
        if reply.strip().lower()!='ok':raise RuntimeError('Stimulus acknowledgement unknown')

    def verify_recording_ready(self):
        if (not self.record_attempted or self.current_file is None or
            self.current_file.name!='stimulus_post.raw.h5' or
            not self.current_file.is_file() or self.current_file.stat().st_size<=0):
            raise RuntimeError('Stimulus file not recording or not present on disk')
        self.progress()
        if (len(self.growth_events)<3 or self.growth_events[-1][0]-self.growth_events[0][0]<20
            or self.clock.monotonic()-self.last_growth>20 or self.last_size<65536):
            raise RuntimeError('Need continuing post-file growth beyond header creation')
        atomic_json(self.out/'presend_growth.json',dict(events=self.growth_events,bytes=self.last_size,
                    limitation='File growth proxy; raw sample continuity still requires the final closed-H5 audit.'))

    def progress(self):
        if shutil.disk_usage(self.out).free<2*1024**3:raise RuntimeError('Low disk space')
        if self.record_attempted:
            size=self.current_file.stat().st_size if self.current_file.exists() else 0
            now=self.clock.monotonic()
            if size>self.last_size:
                self.last_growth=now;self.last_size=size;self.growth_events.append([now,size])
            if now-self.last_growth>120:raise RuntimeError('H5 not growing for 120 s')

    def cleanup(self):
        if hasattr(self,'disarm'):self.disarm()
        self.cleaning=True;errors=[]
        def attempt(label,fn):
            try:fn()
            except BaseException as exc:errors.append(label+': '+repr(exc))
        if self.touched:attempt('outputs off and DAC zero',self.zero)
        s=mx.Saving()
        if self.record_attempted:attempt('stop recording',s.stop_recording)
        if self.file_attempted:attempt('stop file',s.stop_file)
        if self.file_attempted or self.record_attempted:attempt('clear groups',s.group_delete_all)
        for token in self.sequence_tokens:attempt('delete '+token,lambda t=token:self._raw('sequence_delete '+t))
        if self.array_token:attempt('delete array',lambda:self._raw('mea_array_delete '+self.array_token))
        self.record_attempted=False;self.file_attempted=False
        if self.lease:self.lease.close()
        return errors

    def audit(self):
        result=dict(status='incomplete',training_admitted=False)
        path=self.out/'stimulus_post.raw.h5'
        if path.exists():
            timing=self.record_times.get('stimulus_post',{})
            expected=timing.get('end',self.clock.monotonic())-timing.get('start',self.clock.monotonic())
            result=self._inspect('stimulus_post',expected,'post',self.candidate)
        manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(self.out.glob('*.raw.h5'))]
        atomic_json(self.out/'raw_manifest.json',manifest)
        return result
