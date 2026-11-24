"""A prospectively frozen readout panel for one explicitly continued, previously closed study."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import session as station
import numpy as np
from livepair.protocol import RATE
from livepair.readout import read_window
from mea_state2_r2.features import fit
from select_sites import coordinates


def frozen_plan(parent):
    previous=station.load(parent/'paired'/'binding.json')
    previous_close=station.load(parent/'paired'/'closed.json')
    if previous_close['errors'] or previous_close['attempts']!=3:
        raise ValueError('The parent did not close with exactly three localization attempts')
    for label in 'HWF':
        r=station.load(parent/(label+'-result.json'))
        if not r['onset_visible'] or not r['stim_channel_recovered_at_12s']:
            raise ValueError('Incomplete localization evidence')
    path=parent/'paired'/'baseline'/'baseline.raw.h5'
    with station.H5Recording(path,station.CFG) as rec:
        if rec.metadata['chip_id']!=previous['chip_id'] or rec.events or rec.duration<64:
            raise ValueError('A closed, same-chip, unstimulated discovery baseline is required')
        stim=set(previous['electrodes'].values())
        channels=sorted(c for c,e in rec.mapping if e not in stim and c not in previous['bad_channels'])
        segments=[]
        for lo in (0,20,40):
            spikes=rec.spike_channels[(rec.spike_times>=lo)&(rec.spike_times<lo+20)]
            segments.append(np.bincount(spikes,minlength=1024))
        counts=np.array(segments)
        channels=[c for c in channels if np.count_nonzero(counts[:,c])>=2]
        if len(channels)<64:raise ValueError('Fewer than 64 recurrently active independent channels')
        med=[];mad=[];railed=[]
        for t in range(2,63,3):
            a=rec.sample(channels,t)
            center=np.median(a,axis=1)
            med.append(center);mad.append(np.median(abs(a-center[:,None]),axis=1))
            railed.append(np.any((a==0)|(a==1023),axis=1))
        med=np.asarray(med);mad=np.asarray(mad);railed=np.asarray(railed)
        valid=np.zeros(len(channels),dtype=int)
        pairs=0
        for i in range(len(med)):
            for j in range(i+1,len(med)):
                # Symmetric agreement; avoids privileging a particular reference window.
                lo=np.minimum(mad[i],mad[j]);hi=np.maximum(mad[i],mad[j])
                valid+=(lo>0)&(lo>=.67*hi)&(hi<=1.5*lo)&(abs(med[i]-med[j])<=5)&~railed[i]&~railed[j]
                pairs+=1
        ranked=sorted(range(len(channels)),key=lambda i:(-int(valid[i]),float(np.ptp(med[:,i])),
                      -int(np.min(counts[:,channels[i]])),-int(np.sum(counts[:,channels[i]])),channels[i]))
        roi=sorted(channels[i] for i in ranked[:64])
        return dict(previous_session=parent.name,source=rec.source,chip=previous['chip_id'],
                    original_start=previous['started_at'],route_sha256=previous['route_sha256'],
                    eligible=len(channels),panel=roi,comparison_pairs=pairs,
                    min_selected_agreement=min(int(valid[i]) for i in ranked[:64])/pairs,
                    selection='previous quiet recording only; fixed 64, ranked by all 1-s window pairs sampled every 3 s',
                    validation='new quiet recording; at least 50 active and 95% of the fixed 64 at each of 20, 40 and 55 s')


class PanelDevice(station.Device):
    def prime(self,name,initial=False):
        if not initial:return super().prime(name,False)
        self.late(200)
        self.open_record(name)
        time.sleep(65)
        source=self.close_record()['path']
        with self.inspect(source) as rec:
            if rec.events or rec.duration<60:raise ValueError('Incomplete independent quiet recording')
            xy,_=coordinates(rec)
            for site,row in self.selected.items():
                if self.sites[site]!=row['channel'] or xy[row['channel']]!=(row['x_um'],row['y_um']):
                    raise ValueError('Actual site identity or coordinates changed')
            self.roi=list(self.b['fixed_readout_channels'])
            if len(set(self.roi))!=64 or not set(self.roi)<=set(rec.channels.tolist()) or set(self.roi)&set(self.sites.values()):
                raise ValueError('Invalid fixed readout channels in new acquisition')
            active=len(set(self.roi)&set(map(int,rec.spike_channels)))
            checks=[]
            ref=rec.sample(self.roi,3)
            for t in (20,40,55):
                st=station.stability(ref,rec.sample(self.roi,t))
                checks.append(dict(t=t,passed=st['passed'],channels=sum(st['channel_pass']),total=64))
            station.write_once(self.run_dir/'panel-validation.json',dict(roi=self.roi,active=active,
                               windows=checks,source=rec.source,passed=active>=50 and all(r['passed'] for r in checks)))
            if active<50 or not all(r['passed'] for r in checks):raise ValueError('Independent fixed-panel validation failed')
            rows=[read_window(rec,rec.first_frame+i*RATE,self.roi,self.bid,name)['features'] for i in range(5,55)]
            if not any(row[0]>0 for row in rows):raise ValueError('No neural spike activity in baseline windows')
            self.norm=fit(rows)
            station.write_once(self.run_dir/'baseline.json',dict(roi=self.roi,normalization=self.norm,source=rec.source,
                               device_units=self.units,site_channels=self.sites,lsb_mv=self.lsb))
            return [read_window(rec,rec.last_frame+1-i*RATE,self.roi,self.bid,name) for i in (5,4,3,2)]


class Followup(station.Study):
    def open(self,b):
        start=station.binding(b)
        if self.dev or b.get('device_flow')!='braindevice_fixed_panel_v1' or not 0<=time.time()-start<5700:
            raise ValueError('Invalid continuation or expired original window')
        parent=station.DATA/b['parent_session']
        if parent.resolve().parent!=station.DATA or b['session_id']!=parent.name+'-fixed64':
            raise ValueError('Continuation must refer to a single registered parent')
        prior=station.load(parent/'paired'/'binding.json')
        for k in ('chip_id','organoid_id','started_at','device_hostname','acquisition_host','record_root',
                  'device_model','material','route_sha256','route_file','electrodes','bad_channels','replicate'):
            if b[k]!=prior[k]:raise ValueError('Changed original binding: '+k)
        claim=station.load(station.REG/('pair-'+station.digest([b['chip_id'],b['organoid_id']])+'.json'))
        if claim['output']!=str(parent):raise ValueError('Parent registration mismatch')
        panel=station.load(parent/'fixed-readout-plan.json')
        if station.digest(panel)!=b['panel_sha256'] or panel['panel']!=b['fixed_readout_channels']:
            raise ValueError('Panel differs from pre-acquisition registration')
        self.lease=(station.REG/'device.lock').open('a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        close=station.load(parent/'paired'/'closed.json')
        if close['errors'] or close['attempts']!=b['prior_nonzero_attempts'] or close['attempts']!=3:
            raise ValueError('Original attempt count mismatch')
        if station._server_snapshot()[1] or not station.stream_pid():raise RuntimeError('Station busy or stream offline')
        if subprocess.check_output(['ss','-Htn','state','established','sport','=',':5555'],text=True).strip():
            raise RuntimeError('5555 already has a client')
        if shutil.disk_usage(station.DATA).free<383385600000:raise RuntimeError('Insufficient space for planned study')
        signal.setitimer(signal.ITIMER_REAL,max(.01,start+7200-time.time()-120))
        out=station.DATA/b['session_id'];out.mkdir(exist_ok=False)
        station.write_once(out/'binding.json',b)
        selection=station.load(parent/'paired'/'spatial-selection.json')
        station.write_once(out/'spatial-selection.json',selection)
        self.wire=station.Receiver(out);self.wire.thread.start()
        d=self.dev=PanelDevice(b,out);d.selected=selection['suggested']
        d.lsb=d.preflight()['lsb_mv'];d.prepare_sites()
        if d.units!=station.load(parent/'site-plan.json')['joint_units']:raise ValueError('Stimulus route changed')
        windows=d.prime('baseline',True)
        def reserve(event):
            if self.wire.error or not self.wire.count:raise RuntimeError('Missing 5555 return')
            d.reserve(event)
        d.before_sequence_send=reserve
        return dict(binding=d.bid,windows=windows,norm=d.norm,out=str(out))


if __name__=='__main__':
    if sys.argv[1:]==['device']:
        station.Study=Followup
        station.main()
    elif len(sys.argv)==3 and sys.argv[1]=='freeze':
        parent=station.DATA/sys.argv[2]
        if parent.resolve().parent!=station.DATA:raise ValueError('Invalid study path')
        plan=frozen_plan(parent)
        station.write_once(parent/'fixed-readout-plan.json',plan)
        old=station.load(parent/'paired'/'binding.json')
        registered=dict(old,parent_session=parent.name,session_id=parent.name+'-fixed64',fixed_readout_channels=plan['panel'],
                        panel_sha256=station.digest(plan),device_flow='braindevice_fixed_panel_v1',device_launch=str(Path(__file__).resolve()))
        station.write_once(parent/'fixed-readout-binding.json',registered)
        print(json.dumps(dict(eligible=plan['eligible'],channels=len(plan['panel']),agreement=plan['min_selected_agreement'])))
    else:raise SystemExit('Expected device or freeze <closed-parent-session>')
