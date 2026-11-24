"""Explicit supplemental 30-site experiment for the already bound current sample."""
import hashlib,json,os,sys,time,uuid
from pathlib import Path
BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
sys.path.insert(0,str(BASE/'corrected/mea_next_recording'))
from device import MaxLabBackend
from recording_backend import RecordingBackend
from protocol import Deadline,validate_binding,atomic_json,session_key
from quality import parse_mapping
from run import deadline_signals
from brain_stimulation import maxlab as mx
from bounded import batch_commands,drive_batches,available_batches,network_gate
from multi_prepare import append_ack_valid
from multi_protocol import validate_routes

def inspect_zero(path,binding,mapping,routes):
    import h5py,numpy as np
    with h5py.File(path,'r') as f:
        d=f['data_store/data0000'];g=d['groups/routed'];raw=g['raw'];frames=g['frame_nos']
        if f['wellplate/id'][0].decode()!=binding['chip_id']:raise RuntimeError('Fresh chip mismatch')
        if f['wellplate/version'][0].decode()!='MaxOne Single Well MEA':raise RuntimeError('Fresh model mismatch')
        m=sorted((int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:])
        if m!=mapping or float(d['settings/sampling'][0])!=20000:raise RuntimeError('Fresh mapping/rate mismatch')
        n=len(frames)
        if raw.shape!=(1024,n) or raw.dtype.kind not in 'ui' or not 57<=n/20000<=63:raise RuntimeError('Fresh raw shape/duration')
        if not np.array_equal(g['channels'][:].reshape(-1),np.arange(1024)):raise RuntimeError('Unknown raw row order')
        ch=np.array([a for a,b in mapping]);previous=None;max_rail=0.
        for lo in range(0,n,20000):
            block=raw[:,lo:min(n,lo+20000)][ch,:]
            if np.any(block<0) or np.any(block>1023):raise RuntimeError('ADC bounds')
            rail=float(np.max(((block==0)|(block==1023)).mean(axis=1)));max_rail=max(rail,max_rail)
            if rail>.01:raise RuntimeError('Fresh rail excursion')
            fn=frames[lo:min(n,lo+20000)].astype('i8')
            if np.any(np.diff(fn)!=1) or (previous is not None and fn[0]!=previous+1):raise RuntimeError('Fresh frame gap')
            previous=int(fn[-1])
        events=[(int(x['eventid']),int(x['frameno']),int(x['eventtype'])) for x in d['events'][:]]
        if len(events)!=160:raise RuntimeError('Zero batch event count')
        starts=[]
        expected=[b*4000+p*200 for b in range(5) for p in range(3)]+[16408]
        for i in range(10):
            group=events[i*16:(i+1)*16];starts.append(group[0][1])
            if [(x[0],x[2]) for x in group]!=[(900000+j,1) for j in range(16)]:raise RuntimeError('Zero event IDs')
            if [x[1]-group[0][1] for x in group]!=expected:raise RuntimeError('Zero timing/tail mismatch')
        if any(not 19000<=x<=21000 for x in np.diff(starts)):raise RuntimeError('Zero batch cadence')
        sp=d['spikes'][:];start=int(frames[0]);end=int(frames[-1]);valid=(sp['frameno']>=start)&(sp['frameno']<=end)
        eligible=[]
        for r in routes:
            times=np.sort(sp['frameno'][valid&(sp['channel']==r['channel'])])
            if len(times)>=10 and float(np.mean(np.diff(times)<20))<.02:eligible.append(r)
        if len(eligible)<2:raise RuntimeError('Insufficient freshly active sites')
        independent=set(ch.tolist())-{r['channel'] for r in routes}
        active=len(set(sp['channel'][valid].tolist())&independent)
        if active<len(independent)/2:raise RuntimeError('Fresh independent activity rejected')
        return dict(status='passed',duration_s=n/20000,frame_gaps=0,max_rail_fraction=max_rail,
                    zero_batches=10,events=events,independent_channels=len(independent),active_independent_channels=active,
                    eligible_routes=eligible,training_admitted=False,
                    limitation='60-second local switching/recording check; prior10min baseline preceded earlier15pulse exposure; not a new naive baseline or tissue safety validation')

def run():
    import h5py,numpy as np
    binding=json.loads((BASE/'binding.json').read_text());start=validate_binding(binding,time.time())
    out=BASE/'multi_bounded_resume2';journal=Path.home()/'.local/state/mea_photo_full/sessions'
    key=session_key(binding,start);prior=journal/(key+'.stimulus.json')
    old=json.loads((BASE/'recording_resume2/progress.json').read_text())
    if old.get('stage')!='cleanup' or old.get('cleanup_errors') or not old.get('stimulus_acknowledged'):raise RuntimeError('Original capture not safely closed')
    if json.loads((BASE/'recording_resume2/baseline_qc.json').read_text())['status']!='passed':raise RuntimeError('Original baseline not passed')
    with h5py.File(BASE/'recording_resume2/stimulus_post.raw.h5','r') as f:
        data=f['data_store/data0000'];frames=data['groups/routed/frame_nos'];events=data['events'][:]
        if f['wellplate/id'][0].decode()!=binding['chip_id'] or float(data['settings/sampling'][0])!=20000:raise RuntimeError('Prior reference identity/rate')
        if len(events)!=15 or list(events['eventid'])!=list(range(2,17)):raise RuntimeError('Prior reference pulse events')
        if (int(frames[-1])-int(events[-1]['frameno'])-8+1)/20000<1800:raise RuntimeError('Prior reference post is incomplete')
        previous=None
        for lo in range(0,len(frames),1000000):
            block=frames[lo:lo+1000000].astype('i8')
            if np.any(np.diff(block)!=1) or (previous is not None and block[0]!=previous+1):raise RuntimeError('Prior reference frame gap')
            previous=int(block[-1])
    prior_bytes=prior.read_bytes();prior_dose=json.loads(prior_bytes)
    if prior_dose['reserved_pulses']!=15:raise RuntimeError('Prior dose binding mismatch')
    # Explicit user amendment: shorter train as needed and 600 s post observation.
    post_seconds=600;reserve_seconds=post_seconds+207
    deadline=Deadline(start,time)
    anchor=old['stages'][0]
    deadline.mono_end=min(deadline.mono_end,anchor['mono']+(start+7200-anchor['wall']))
    deadline.require(post_seconds+318)
    failed=BASE/'multi_bounded_recording'
    failed_report=json.loads((failed/'progress.json').read_text())
    failed_log=[json.loads(x) for x in (failed/'api_audit.jsonl').read_text().splitlines()]
    if failed_report.get('true_attempted') is not False or any(x.get('command','').startswith('sequence_send ') for x in failed_log):raise RuntimeError('Previous attempt may have sent output; cannot recover')
    first_error=next(x for x in failed_log if 'error' in x)
    if first_error['command']!='saving_set_assay_input protocol multi_bounded600_zero_qualification':raise RuntimeError('Unrecognized pre-output failure')
    restored=json.loads((BASE/'service_restore.json').read_text())
    if restored.get('status')!='passed' or not restored.get('all32_off_and_dac512_acknowledged'):raise RuntimeError('Service not safely restored')
    if (journal/(key+'.multi_bounded600.stimulus.json')).exists():raise RuntimeError('Additional dose already consumed')
    zero_failed=BASE/'multi_bounded_resume1'
    zero_report=json.loads((zero_failed/'progress.json').read_text())
    zero_log=[json.loads(x) for x in (zero_failed/'api_audit.jsonl').read_text().splitlines()]
    if zero_report.get('error')!="RuntimeError('Fresh rail excursion')" or zero_report.get('true_attempted') is not False or zero_report.get('cleanup_errors'):
        raise RuntimeError('Unexpected zero-test recovery state')
    sends=[x['command'] for x in zero_log if x.get('command','').startswith('sequence_send ')]
    if not sends or any(not cmd.startswith('sequence_send bounded_zero_') for cmd in sends):raise RuntimeError('Previous supplemental output cannot be excluded')
    if (zero_failed/'true_preview.json').exists():raise RuntimeError('True profile previously prepared')
    original_launch=json.loads((journal/(key+'.multi_bounded600.launch.json')).read_text())
    if original_launch['binding']!=binding:raise RuntimeError('Recovery binding changed')
    atomic_json(journal/(key+'.multi_bounded600.stable_zero_resume2.launch.json'),dict(binding=binding,prior_claim_sha256=hashlib.sha256(prior_bytes).hexdigest(),
        user_amendment='up_to32_as_long_as_window_allows_max600seconds_post600seconds',restored_server_pid=restored['server_pid'],output=str(out)),exclusive=True)
    out.mkdir(exist_ok=False);(out/'slots').mkdir()
    atomic_json(out/'supplement_source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    atomic_json(out/'binding.json',dict(binding=binding,post_seconds=post_seconds,
        user_authorized_shorter_train=True,user_authorized_short_post=True,prior_single_site_pulses=15,
        original_deadline_wall=start+7200,original_deadline_mono=deadline.mono_end))
    d=RecordingBackend(binding,out)
    report=dict(status='preparing',alpha=1,r2_mode='record_only',realtime_training=False,offline_training=False,
                training_admitted=False,prior_single_site_pulses=15,true_attempted=False,post_seconds=post_seconds,cleanup_errors=[])
    d.controller=type('Controller',(),{'deadline':deadline})()
    def bounded_zero():
        previous=d.cleaning;d.cleaning=True
        try:d.zero()
        finally:d.cleaning=previous
    def save(stage=None):
        if stage:report['stage']=stage
        report['wall']=time.time();atomic_json(out/'progress.json',report)
    def wait(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            deadline.require(120);d.progress();save();time.sleep(min(1,max(0,end-time.monotonic())))
    d.wait=wait
    def upload(preview,token):
        d.sequence_tokens.append(token)
        if d._raw('sequence_new '+token).strip()!=token:raise RuntimeError('Sequence token')
        for cmd in preview['commands']:
            if not append_ack_valid(cmd,d._raw('sequence_append '+token+'\n'+cmd)):raise RuntimeError('Append acknowledgement')
    try:
        with d.sdk_context(),deadline_signals(start+7200-time.time()-120,d):
            d.preflight();d.touched=True;d.zero()
            mx.initialize();d.zero();mx.send(mx.Core().enable_stimulation_power(True))
            d.array_token='bounded_route_'+uuid.uuid4().hex;d.array=mx.Array(d.array_token,persistent=True);d.array.load_config_data(d.route)
            for amp in d.active_links().values():d.array.disconnect_amplifier_from_stimulation(amp)
            if d.active_links():raise RuntimeError('Inherited links')
            routes=json.loads((BASE/'bounded_routes.json').read_text())['routes'];expected={}
            for r in routes:
                d.array.connect_electrode_to_stimulation(r['electrode']);expected[r['unit']]=r['channel']
                if d.active_links()!=expected:raise RuntimeError('Frozen route changed')
            validate_routes(routes,d.active_links());mx.activate([0]);d.array.download([0]);wait(5)
            cfg=d._raw('mea_array_command '+d.array_token+'\nmea_array_get_config')
            if parse_mapping(cfg)!=d.mapping:raise RuntimeError('Downloaded recording map changed')
            mx.offset();wait(10);d.zero();wait(5);mx.clear_events()
            lsb=float(mx.query_DAC_lsb_mV());units=[r['unit'] for r in routes]
            zero=batch_commands(lsb,units,zero=True,switch_units=False);zt='bounded_zero_'+uuid.uuid4().hex;upload(zero,zt)
            atomic_json(out/'zero_preview.json',zero);atomic_json(out/'connected_routes.json',routes)
            for unit in units:
                command=mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(0).set()
                if d._raw(command).strip().lower()!='ok':raise RuntimeError('Stable unit enable not acknowledged')
            report['batch_tail_guarantee']='DAC512 only; units remain powered between batches; explicit all32-off on exit'
            save('stable_connection_zero_settle_20s');wait(20)
            d.open('identity');save('zero_qualification_60s')
            begin=time.monotonic()
            def zsend(i):
                if d._raw('sequence_send '+zt).strip().lower()!='ok':raise RuntimeError('Zero send unknown')
            zr=drive_batches(time,lambda i:None,zsend,d.progress,batch_count=10)
            atomic_json(out/'zero_batches.json',zr)
            if zr['status']!='completed':raise RuntimeError('Zero cadence check failed')
            wait(max(0,60-(time.monotonic()-begin)));d.close();save('zero_qc')
            q=inspect_zero(out/'identity.raw.h5',binding,d.mapping,routes);atomic_json(out/'zero_qc.json',q)
            routes=q['eligible_routes'];units=[r['unit'] for r in routes]
            for unit in set(range(32))-set(units):
                if d._raw(mx.StimulationUnit(unit).power_up(False).connect(False).set()).strip().lower()!='ok':raise RuntimeError('Excluded unit shutdown not acknowledged')
            true=batch_commands(lsb,units,zero=False,switch_units=False);tt='bounded_true_'+uuid.uuid4().hex;upload(true,tt)
            atomic_json(out/'true_preview.json',true);atomic_json(out/'stimulated_routes.json',routes)
            report['site_count']=len(routes)
            d.open('stimulus_post');save('true_file_leadin_30s');wait(30)
            d.verify_recording_ready();planned=available_batches(deadline.remaining(),post_seconds=post_seconds);report['planned_batches']=planned
            atomic_json(journal/(key+'.multi_bounded600.stimulus.json'),dict(binding=binding,max_batches=planned,pulses_per_batch=15,
                site_count=len(routes),routes=routes,post_seconds=post_seconds,nominal_phase_mv=true['nominal_phase_mv'],prior_claim_sha256=hashlib.sha256(prior_bytes).hexdigest(),
                no_retry=True,wall=time.time(),output=str(out)),exclusive=True)
            save('true_batches');trainstart=time.monotonic();network=d.request;slot_state={'index':None}
            def checked_network(message,timeout):
                if message=='sequence_send '+tt:
                    if slot_state['index'] is None:raise RuntimeError('Missing reserved batch')
                    network_gate(time.monotonic(),trainstart,slot_state['index'])
                    deadline.require(max(0,planned-(time.monotonic()-trainstart))+reserve_seconds)
                return network(message,timeout)
            d.request=checked_network
            def reserve(i):
                atomic_json(out/'slots'/f'{i:03}.json',dict(batch=i,wall=time.time(),mono=time.monotonic(),consumed=True),exclusive=True)
                report['true_attempted']=True;report['last_reserved_batch']=i
                slot_state['index']=i
                if i%10==0:save()
            def health():
                deadline.require(max(0,planned-(time.monotonic()-trainstart))+reserve_seconds);d.progress()
            def send(i):
                report['true_attempted']=True;report['last_batch']=i
                if d._raw('sequence_send '+tt).strip().lower()!='ok':raise RuntimeError('True acknowledgement unknown')
            try:result=drive_batches(time,reserve,send,health,batch_count=planned,start_mono=trainstart)
            except BaseException as exc:result=dict(status='output_unknown',error=repr(exc))
            report['batch_result']=result;atomic_json(out/'batch_result.json',result)
            try:bounded_zero()
            except BaseException as off_error:report['cleanup_errors'].append('post-train zero: '+repr(off_error))
            wait(1);report['post_end_mono']=time.monotonic()+post_seconds;save('post_600s');wait(max(0,report['post_end_mono']-time.monotonic()));d.close()
            report['status']='recording_finished_pending_audit' if result['status']=='completed' else 'stopped_with_recovery'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        # Keep observation after any dose attempt if the recording path survives.
        if report['true_attempted'] and d.record_attempted:
            try:
                with d.sdk_context():
                    try:bounded_zero()
                    except BaseException as off_error:report['cleanup_errors'].append('error zero: '+repr(off_error))
                    wait(1)
                    if 'post_end_mono' not in report:report['post_end_mono']=time.monotonic()+post_seconds
                    save('error_recovery_remaining');wait(max(0,report['post_end_mono']-time.monotonic()));d.close()
            except BaseException as recovery_error:report['recovery_error']=repr(recovery_error)
    finally:
        with d.sdk_context():report['cleanup_errors'].extend(d.cleanup())
        save('cleanup')
    print(json.dumps(report,indent=2));return 0 if report['status']=='recording_finished_pending_audit' and not report['cleanup_errors'] else 1

if __name__=='__main__':
    if sys.argv[1:]!=['--run']:raise SystemExit('Explicit --run required; no default execution')
    raise SystemExit(run())
