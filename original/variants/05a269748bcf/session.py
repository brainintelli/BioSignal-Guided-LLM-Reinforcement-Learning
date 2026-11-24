"""Explicit stages with a durable single-send boundary and no training path."""
from pathlib import Path
from protocol import (validate_binding, Deadline, atomic_json, session_key,
                      SEND_RESERVE, SEQUENCE_SECONDS, POST_SECONDS, CLOSEOUT_SECONDS)


class Session:
    def __init__(self,binding,backend,clock,output,journal):
        self.binding=binding; self.device=backend; self.clock=clock
        self.output=Path(output); self.journal=Path(journal)
        self.device.controller=self;self.device.wait=self.wait
        self.report=dict(status='starting',alpha=1,r2_mode='record_only',
                         realtime_training=False,offline_training=False,
                         training_admitted=False,stimulus_attempted=False,
                         stimulus_acknowledged=False,cleanup_errors=[],stages=[])

    def save(self):
        self.report['wall_time']=self.clock.time()
        atomic_json(self.output/'progress.json',self.report)

    def stage(self,name):
        self.report['stage']=name
        self.report['stages'].append(dict(name=name,wall=self.clock.time(),mono=self.clock.monotonic()))
        self.save()

    def wait(self,seconds):
        self.deadline.require(seconds+CLOSEOUT_SECONDS)
        end=self.clock.monotonic()+seconds
        while self.clock.monotonic()<end:
            self.deadline.require(CLOSEOUT_SECONDS)
            self.device.progress()
            left=end-self.clock.monotonic()
            self.report['phase_remaining_seconds']=max(0,left);self.save()
            self.clock.sleep(min(10,max(0,left)))

    def run(self):
        start=validate_binding(self.binding,self.clock.time())
        self.deadline=Deadline(start,self.clock)
        # 60 identity + 10 settling + 20 route/settle + baseline + 60 lead-in
        # + full post and closeout. QC/upload time is checked again, not guessed away.
        self.deadline.require(60+10+20+600+60+SEND_RESERVE)
        self.journal.mkdir(parents=True,exist_ok=True)
        key=session_key(self.binding,start)
        atomic_json(self.journal/(key+'.launch.json'),
                    dict(binding=self.binding,output=str(self.output),started_wall=self.clock.time()),
                    exclusive=True)
        self.output.mkdir(parents=True,exist_ok=False)
        atomic_json(self.output/'binding.json',self.binding,exclusive=True)
        self.save()
        try:
            self.stage('preflight'); self.device.preflight()
            self.stage('identity_setup'); self.device.setup_identity()
            self.stage('identity_settling'); self.wait(10)
            self.stage('fresh_identity_60s'); self.device.open('identity'); self.wait(60); self.device.close()
            self.report['candidate']=self.device.identity(); self.save()
            self.stage('configure_current_route'); self.device.configure(self.report['candidate'])
            # Adapter calls the controller wait for download and offset ordering.
            self.stage('connection_settling'); self.wait(5)
            self.device.prepare()
            self.stage('baseline_600s'); self.device.open('baseline');self.device.marker()
            self.wait(600); self.device.close()
            self.stage('baseline_qc'); self.report['baseline_qc']=self.device.qualify(); self.save()
            if self.report['baseline_qc'].get('status')!='passed': raise RuntimeError('Baseline quality gate rejected')
            self.deadline.require(60+SEND_RESERVE)
            self.stage('post_file_leadin_60s'); self.device.open('stimulus_post'); self.wait(60)
            self.deadline.require(SEND_RESERVE)
            atomic_json(self.journal/(key+'.stimulus.json'),
                        dict(reserved_pulses=15,max_sends=1,wall=self.clock.time(),
                             output=str(self.output),no_retry=True),exclusive=True)
            self.report['stimulus_attempted']=True
            self.stage('single_stimulus_attempt')
            try:
                self.device.send();self.report['stimulus_acknowledged']=True
            except Exception as exc:
                self.report['send_error']=repr(exc)
                self.report['status']='output_unknown'
            # Even a lost reply consumes the sequence, and gets recovery recording.
            self.wait(SEQUENCE_SECONDS)
            self.stage('post_1800s')
            post_end=self.clock.monotonic()+POST_SECONDS
            try: self.device.zero()
            except Exception as exc: self.report['cleanup_errors'].append('early zero: '+repr(exc))
            self.wait(max(0,post_end-self.clock.monotonic()));self.device.close()
            if self.report['status']!='output_unknown': self.report['status']='completed'
        except BaseException as exc:
            self.report.update(status='failed',error=repr(exc))
        finally:
            try:self.stage('cleanup')
            except BaseException as exc:self.report['cleanup_errors'].append('cleanup log: '+repr(exc))
            try: self.report['cleanup_errors'].extend(self.device.cleanup())
            except BaseException as exc: self.report['cleanup_errors'].append(repr(exc))
            if self.report['cleanup_errors']: self.report['status']='failed'
            try:self.save()
            except OSError as exc:self.report['metadata_error']=repr(exc)
        # Audit after all output and recording resources have been closed.
        try:
            self.report['audit']=self.device.audit()
            if self.report['audit'].get('status')!='passed' and self.report['status']=='completed':
                self.report['status']='audit_failed'
        except BaseException as exc:
            self.report['audit_error']=repr(exc)
            if self.report['status']=='completed': self.report['status']='audit_failed'
        try:self.save()
        except OSError as exc:self.report.update(status='failed',metadata_error=repr(exc))
        return self.report
