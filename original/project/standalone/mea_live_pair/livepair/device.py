"""Single-owner acquisition controller for the three bound, individually routed sites."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

import numpy as np
from mea_binary_r2.encoding import Intent, compile_sequence
from mea_state2_r2.features import digest, fit
from mea_stim_v2.config import route_mapping
from mea_stim_v2.live import MaxLabBackend, mx
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording
from .protocol import PROFILE, RATE, binding
from .readout import admit, checked_metadata, read_window
from . import predictability


def write_once(path, data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,allow_nan=False,ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


class Route:
    def __init__(self,b):
        self.b=b
        self.device_hostname=b['device_hostname']
        self.route_file=b['route_file']
        self.session_id=b['session_id']
        self.threshold_h5_dataset=b['threshold_h5_dataset']
        self.sites=b['electrodes']
        if set(self.sites)!=set('HWF') or len(set(self.sites.values()))!=3:
            raise ValueError('Three distinct current electrodes required')
        self.stim_electrode=self.sites['H']
        self.validate_binding()

    def validate_binding(self):
        if hashlib.sha256(Path(self.route_file).read_bytes()).hexdigest()!=self.b['route_sha256']:
            raise ValueError('Frozen route changed')
        pairs=route_mapping(self.route_file)
        if not set(self.sites.values())<=set(dict(pairs).values()):
            raise ValueError('Selected electrode absent from route')
        return pairs

    def validate_live(self):
        binding(self.b)
        if self.b['device_model']!='MaxOne' or self.b['material']!='Pt-Black':
            raise ValueError('Unsupported chip')


class Device(MaxLabBackend):
    def __init__(self,b,run_dir):
        self.b=b
        self.start=binding(b)
        self.deadline=time.monotonic()+7200-(time.time()-self.start)
        self.late(180)
        addresses=subprocess.check_output(['hostname','-I'],text=True).split()
        if b['acquisition_host'] not in addresses or socket.gethostname()!=b['device_hostname']:
            raise RuntimeError('Wrong acquisition host')
        super().__init__(Route(b),run_dir,SimpleNamespace(wall=time.time,mono=time.monotonic,sleep=time.sleep))
        self.bid=digest(b)
        self.units={}
        self.last_send=None
        self.last_frame=None
        self.roi=[]
        self.qualified=set()
        self.response_passed=False
        self.attempts=b['prior_nonzero_attempts']
        if type(self.attempts) is not int or not 0<=self.attempts<=30:
            raise ValueError('Prior attempts must include calibration and ambiguous output')
        self.used=set()
        self.serial=0
        self.before_sequence_send=self.reserve

    def late(self,reserve):
        age=time.time()-self.start
        if age<0 or age+reserve>=7200 or time.monotonic()+reserve>=self.deadline:
            raise TimeoutError('Biological time budget exhausted')

    def verify_selection(self):
        from select_sites import candidates
        folder=self.run_dir.with_name(self.b['session_id']+'-site-selection')
        identity=json.loads((folder/'identity.json').read_text())
        for key in ('session_id','organoid_id','chip_id','acquisition_host','device_hostname','started_at','bad_channels'):
            if identity.get(key)!=self.b.get(key):
                raise ValueError('Quiet site-selection registration differs: '+key)
        with H5Recording(folder/'quiet.raw.h5',self.config) as rec:
            checked_metadata(rec,self.b,self.config.validate_binding())
            result=candidates(rec,self.b['chip_id'],self.b['bad_channels'])
        if result!=json.loads((folder/'site-candidates.json').read_text()):
            raise ValueError('Spatial selection does not reproduce on the bound quiet recording')
        self.selected=result['suggested']
        if {s:row['electrode'] for s,row in self.selected.items()}!=self.config.sites:
            raise ValueError('Bound electrodes differ from current spatial selection')
        write_once(self.run_dir/'spatial-selection.json',result)

    def claim(self):
        import fcntl
        registry=Path.home()/'.local/state/mea_photo_full'
        registry.mkdir(parents=True,exist_ok=True)
        self.lease=(registry/'device.lock').open('a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Refuse automatic restart of a sample after an interrupted or completed run.
        key=digest([self.b['chip_id'],self.b['organoid_id']])
        write_once(registry/('pair-'+key+'.json'),dict(binding=self.b,output=str(self.run_dir)))
        write_once(self.run_dir/'binding.json',self.b)
        if shutil.disk_usage(self.run_dir).free<383385600000:
            raise RuntimeError('Insufficient space for raw acquisition')
        self.lsb=self.preflight()['lsb_mv']
        with self._sdk():
            if mx.send_raw('wellplate_query_version').strip()!='0':
                raise RuntimeError('Ordinary MaxOne required')

    def _check_mapping(self):
        raw=mx.send_raw('mea_array_command '+self.array_token+'\nmea_array_get_config')
        if sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',raw.split()[0]))!=self.config.validate_binding():
            raise ValueError('Downloaded route differs')
        channels=dict((e,c) for c,e in self.config.validate_binding())
        links={}
        for unit in range(32):
            value=self.array.query_amplifier_at_stimulation(unit).strip()
            if value not in ('','-1'):
                links[unit]=int(value)
        if links!={unit:channels[self.config.sites[k]] for k,unit in self.units.items()}:
            raise ValueError('Unexpected stimulation attachment')
        for k,unit in self.units.items():
            if int(self.array.query_stimulation_at_electrode(self.config.sites[k]))!=unit:
                raise ValueError('Electrode/unit association changed')
        return raw

    def prepare_sites(self):
        self.late(180)
        self._check_server(False)
        with self._sdk():
            self.touched=True
            mx.initialize()
            for u in range(32):
                mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
            mx.set_gain(512)
            mx.hpf('300Hz')
            mx.send(mx.Core().enable_stimulation_power(True))
            self.array_token='pair_'+uuid.uuid4().hex
            self.array=mx.Array(self.array_token,persistent=True)
            self.array.load_config(self.config.route_file)
            for unit in range(32):
                v=self.array.query_amplifier_at_stimulation(unit).strip()
                if v not in ('','-1'):
                    self.array.disconnect_amplifier_from_stimulation(int(v))
            for key,e in self.config.sites.items():
                self.array.connect_electrode_to_stimulation(e)
                self.units[key]=int(self.array.query_stimulation_at_electrode(e))
            if len(set(self.units.values()))!=3 or any(not 0<=u<32 for u in self.units.values()):
                raise ValueError('Cannot route three independent units')
            mx.activate([0])
            self.array.download([0])
            time.sleep(5)
            mx.offset()
            time.sleep(5)
            raw=self._check_mapping()
        (self.run_dir/'downloaded.cfg').write_text(raw)
        self.prepared=True
        by_e=dict((e,c) for c,e in self.config.validate_binding())
        self.sites={k:by_e[e] for k,e in self.config.sites.items()}

    def reserve(self, event):
        self.late(1500)
        if self.attempts>=30 or (self.last_send is not None and time.monotonic()-self.last_send<20):
            raise RuntimeError('Shared dose or cycle limit')
        self.attempts+=1
        write_once(self.run_dir/f'attempt-{self.attempts:02d}.json',dict(event=event,time=time.time()))

    def inspect(self,path):
        rec=H5Recording(path,self.config)
        try:
            checked_metadata(rec,self.b,self.config.validate_binding())
            if self.last_frame is not None and rec.first_frame<=self.last_frame:
                raise ValueError('Device frame clock regressed between segments')
            self.last_frame=rec.last_frame
            with Path(path).open('rb') as f:
                h=hashlib.sha256()
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            rec.source['sha256']=h.hexdigest()
            return rec
        except BaseException:
            rec.close()
            raise

    def prime(self,name,initial=False):
        self.late(200)
        self.open_record(name)
        time.sleep(65 if initial else 9)
        p=self.close_record()['path']
        with self.inspect(p) as rec:
            if rec.events or rec.duration < (60 if initial else 7):
                raise ValueError('Priming block incomplete or contains a stimulus')
            if initial:
                from select_sites import coordinates
                xy,_=coordinates(rec)
                for site,row in self.selected.items():
                    if self.sites[site]!=row['channel'] or xy[row['channel']]!=(row['x_um'],row['y_um']):
                        raise ValueError('Connected baseline has different physical site geometry')
                excluded=set(self.sites.values())|set(self.b['bad_channels'])
                active=set(int(x) for x in np.unique(rec.spike_channels))
                self.roi=sorted((set(rec.channels.tolist())&active)-excluded)
                if len(self.roi)<50:raise ValueError('Insufficient active independent ROI')
                for offset in (20,40,55):
                    if not stability(rec.sample(self.roi,3),rec.sample(self.roi,offset))['passed']:
                        raise ValueError('Baseline noise unstable')
                rows=[read_window(rec,rec.first_frame+i*RATE,self.roi,self.bid,name)['features'] for i in range(5,55)]
                if not any(row[0]>0 for row in rows):raise ValueError('No native spike activity')
                self.norm=fit(rows)
                write_once(self.run_dir/'baseline.json',dict(roi=self.roi,normalization=self.norm,source=rec.source,
                    device_units=self.units,site_channels=self.sites,lsb_mv=self.lsb))
            return [read_window(rec,rec.last_frame+1-i*RATE,self.roi,self.bid,name) for i in (5,4,3,2)]

    def slot(self,req,calibration=False):
        self.late(180)
        if not calibration and not self.response_passed:
            raise ValueError('Current sample has not passed the three-site response test')
        n=req['pulses']
        predictive = not calibration and self.b.get('feedback_encoding') == predictability.MODE
        if predictive:
            events = predictability.verify_request(req)
        elif 'encoding' in req:
            raise ValueError('Feedback mode not registered')
        if (req['id'] in self.used or req['id']!=digest({k:v for k,v in req.items() if k!='id'})
            or req['binding']!=self.bid or req['profile']!=PROFILE.digest or n not in (0,3)
            or (not predictive and ((n and req['site'] not in self.sites) or (not n and req['site'] is not None)))
            or (not calibration and self.qualified!=set('HWF'))):
            raise ValueError('Invalid, repeated or unqualified physical request')
        self.used.add(req['id'])
        self.serial+=1
        eid=self.serial*4
        if self.last_send is not None:
            time.sleep(max(0.,20.1-(time.monotonic()-self.last_send)))
        self.open_record('slot_%03d'%self.serial)
        time.sleep(4)
        size=self.record_path.stat().st_size if self.record_path.is_file() else 0
        for _ in range(30):
            time.sleep(1)
            if self.record_path.is_file() and self.record_path.stat().st_size>max(65536,size):break
        else:
            raise RuntimeError('No raw recording growth before output')
        self.assert_ready()
        if not self.record_path.is_file() or self.record_path.stat().st_size<=0:
            raise RuntimeError('Raw file absent before output')
        with self._sdk():
            if float(mx.query_DAC_lsb_mV())!=self.lsb:
                raise RuntimeError('DAC readback changed')
            if n:self.late(1500)
            mx.send(mx.DAC(0,512))
            if predictive:
                for dac in (1,2):
                    mx.send(mx.DAC(dac,512))
            for key,u in self.units.items():
                enabled = any(k==key for k,_ in events) if predictive else bool(n and key==req['site'])
                dac = predictability.SITES.index(key) if predictive else 0
                mx.send(mx.StimulationUnit(u).power_up(enabled).connect(enabled).set_voltage_mode().dac_source(dac))
            codes=[]
            marker=dict(tx=req['id'],idx=0)
            if predictive:
                marker['site']=events[0][0] if n else 'quiet'
                for kind,value in predictability.sequence(req,self.lsb):
                    if kind=='event_index':
                        site=events[value][0]
                        msg=f'tx {req["id"]} idx {value} site {site}'
                        codes.append(('marker',mx.Event(0,1,eid+value,msg)))
                    else:
                        codes.append((kind,value))
            else:
                intent=Intent(req['site'],n,True,(req['site'],) if n else ())
                onset=0
                for k,v in compile_sequence(intent,self.lsb,PROFILE):
                    if k=='dac' and v<512:
                        if onset:
                            codes.append(('marker',mx.Event(0,1,eid+onset,f'tx {req["id"]} idx {onset}')))
                        onset+=1
                    codes.append((k,v))
            self._sequence(eid,marker,codes)
            self.last_send=time.monotonic()
            time.sleep(.45 if predictive and n else .05)
            for u in self.units.values():
                mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
            mx.send(mx.DAC(0,512))
        time.sleep(14)
        p=self.close_record()['path']
        with self.inspect(p) as rec:
            result=admit(rec,req,eid,self.roi,self.sites)
        write_once(self.run_dir/f'receipt-{self.serial:03d}.json',result)
        if calibration and n:self.qualified.add(req['site'])
        return result

    def verify_output_calibration(self):
        """Load measurements are distinct from the ADC artifact admission on each slot."""
        cal=self.read_calibration()
        if cal['dac_lsb_mv']!=self.lsb or cal['profile']!=PROFILE.digest or set(cal['measurements'])!=set('HWF'):
            raise ValueError('Calibration waveform differs')
        for k,m in cal['measurements'].items():
            if not (m['polarity']=='positive_first' and 0<m['positive_peak_mv']<=200
                    and -200<=m['negative_peak_mv']<0 and m['unit']==self.units[k]):
                raise ValueError('Measured phase/route outside bound profile')
        if self.b.get('feedback_encoding') == predictability.MODE:
            # Qualification uses DAC0 for all sites. Task feedback uses three
            # independent DACs. Require measurements of both routed configurations.
            routed=cal.get('feedback_measurements',{})
            if set(routed)!=set(predictability.SITES):
                raise ValueError('Missing load calibration of the three feedback DAC routes')
            for dac,k in enumerate(predictability.SITES):
                m=routed[k]
                if (m['dac']!=dac or m['unit']!=self.units[k] or m['polarity']!='positive_first'
                        or not 0<m['positive_peak_mv']<=200 or not -200<=m['negative_peak_mv']<0):
                    raise ValueError('Incorrect feedback load measurement')
        write_once(self.run_dir/'calibration-evidence.json',cal)

    def read_calibration(self):
        p=Path(self.b['calibration_file'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=self.b['calibration_sha256']:
            raise ValueError('Calibration record differs')
        cal=json.loads(p.read_text())
        if any(cal[k]!=self.b[k] for k in ('session_id','chip_id','route_sha256','electrodes')):
            raise ValueError('Calibration identity differs')
        for m in (*cal['measurements'].values(), *cal.get('feedback_measurements',{}).values()):
            src=p.parent/m['trace_file']
            if not src.stat().st_size or hashlib.sha256(src.read_bytes()).hexdigest()!=m['trace_sha256']:
                raise ValueError('Load measurement evidence missing')
        return cal
