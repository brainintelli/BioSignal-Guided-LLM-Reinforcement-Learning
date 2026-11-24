"""Separate raw progress, receive-clock measurements, and unknown event coverage."""
import math
from .mea_protocol import integer, number, strict_json, _Lines


class FilteredFrameStream(_Lines):
    """Transport-frame prefix gate; its watermark is NOT a spike-peak watermark."""
    def __init__(self):
        super().__init__(2_000_000)
        self.seq=-1;self.session_id=None;self._watermark=None;self.start_frame=None
        self.frame_count=0;self.event_count=0;self.quiet_blocks=0;self.pending=[];self.ended=False
        self.spike_keys=set()

    @property
    def watermark(self):return None if self.failed else self._watermark

    def _line(self,line):
        row=strict_json(line)
        if not isinstance(row,dict):raise ValueError('frame packet must be an object')
        if self.ended or integer(row.get('seq'),'sequence')!=self.seq+1:raise ValueError('frame stream sequence/end')
        kind=row.get('type')
        if self.seq<0:
            if kind!='session' or row.get('protocol')!='mea.filtered.transport.v1' or row.get('sdk_open_status')!=0:
                raise ValueError('filtered frame session')
            if row.get('progress_timebase')!='transport_frame' or row.get('event_timebase')!='spike_peak_frame':
                raise ValueError('explicit frame timebases required')
            if not isinstance(row.get('session_id'),str) or not row['session_id']:raise ValueError('session identity')
            self.session_id=row['session_id']
        elif kind=='events':
            frame=integer(row['frame_number'],'filtered event frame')
            if self.watermark is not None and frame<self.watermark:raise ValueError('late event after frame barrier')
            if not isinstance(row['events'],list):raise ValueError('events array')
            for event in row['events']:
                if len(event)!=4 or integer(event[0],'event frame')>frame or integer(event[2],'well')!=0:
                    raise ValueError('spike/frame association')
                integer(event[1],'channel');number(event[3],'amplitude')
                if self.pending and frame<self.pending[-1][0]:raise ValueError('event order')
                key=(event[0],event[1])
                if key in self.spike_keys or len(self.spike_keys)>=5_000_000:raise ValueError('duplicate spike/capacity')
                self.spike_keys.add(key)
                self.pending.append((frame,event[1],event[0]));self.event_count+=1
                if len(self.pending)>1_024_000:raise ValueError('pending event capacity')
        elif kind=='frame_heartbeat':
            integer(row['mono_ns'],'heartbeat time');integer(row['since_frame_ns'],'frame age')
            if not self.frame_count<=integer(row['frame_count'],'frames')<self.frame_count+1000:
                raise ValueError('heartbeat frame count')
        elif kind=='frame_progress':
            start=integer(row['start_frame'],'start');end=integer(row['end_frame'],'end');count=integer(row['count'],'count',1)
            if count>1000 or end-start!=count or (self.watermark is not None and start!=self.watermark):
                raise ValueError('filtered frame discontinuity')
            if row['complete'] is not True or row['discontinuities']!=0 or row['corrupted']!=0:
                raise ValueError('invalid filtered coverage')
            if row['cumulative_frames']!=self.frame_count+count or row['event_count']!=self.event_count:
                raise ValueError('frame/event count barrier')
            if any(not start<=e[0]<end for e in self.pending):raise ValueError('event outside frame block')
            if not self.pending:self.quiet_blocks+=1
            self.pending.clear();self.frame_count+=count
            if self.start_frame is None:self.start_frame=start
            self._watermark=end
        elif kind=='end':
            if row.get('clean_end') is not True or row.get('sdk_close_status')!=0 or row.get('sdk_errors')!=0 or row.get('event_count')!=self.event_count or row.get('received_frames')!=self.frame_count or self.pending or self.watermark is None:
                raise ValueError('unclean filtered stream end')
            self.ended=True
        else:raise ValueError('unknown filtered frame packet')
        self.seq=row['seq'];return row

    def finish(self):
        if not self.ended:
            self.failed=True;raise ValueError('missing filtered stream footer')
        super().finish()


def audit_filtered_progress(rows,h5_frames,sample_rate_hz,event_audit_passed):
    import copy
    from .mea_protocol import canonical
    gate=FilteredFrameStream()
    for row in rows:gate.feed((canonical(row)+'\n').encode())
    gate.finish()
    session=copy.deepcopy(rows[0]);session.update(protocol='mea.raw_progress.v1',seq=0)
    normalized=[session]
    for row in rows:
        if row.get('type')=='frame_heartbeat':
            normalized.append(dict(type='heartbeat',seq=len(normalized),raw_frames=row['frame_count'],
                                   since_raw_ns=row['since_frame_ns'],mono_ns=row['mono_ns']))
            continue
        if row.get('type')!='frame_progress':continue
        block=copy.deepcopy(row);block.update(type='raw_block',seq=len(normalized),filtered_events=row['event_count'])
        normalized.append(block)
    footer=copy.deepcopy(rows[-1]);footer.update(seq=len(normalized),raw_frames=gate.frame_count,raw_contiguous=True,errors=footer['sdk_errors'])
    normalized.append(footer)
    report=audit_progress(normalized,h5_frames,sample_rate_hz,[])
    for old,new in [('raw_progress_passed','filtered_progress_passed'),('raw_freshness_passed','filtered_freshness_passed'),
                    ('raw_interval','filtered_interval'),('raw_frames','filtered_frames')]:
        if old in report:report[new]=report.pop(old)
    report.update(schema='mea.filtered_timing.audit.v1',quiet_blocks=gate.quiet_blocks,
                  covered_start_frame=gate.start_frame,covered_end_frame=gate.watermark,
                  filtered_frame_prefix_verified=report['filtered_progress_passed'],
                  recorded_event_agreement_passed=event_audit_passed)
    report['transport_frame_windows_complete']=report['filtered_progress_passed'] and event_audit_passed is True
    report['live_spike_window_complete']=False
    report['spike_peak_window_complete']=False
    report['observed_transport_gate_passed']=all(report[k] for k in
        ('filtered_progress_passed','filtered_freshness_passed','receive_clock_fit_passed','wall_clock_stable','recorded_event_agreement_passed'))
    offsets=[row['frame_number']-event[0] for row in rows if row.get('type')=='events' for event in row['events']]
    report['observed_transport_minus_peak_frames']=dict(count=len(offsets),min=min(offsets) if offsets else None,
        p50=percentile(offsets,.5),p95=percentile(offsets,.95),max=max(offsets) if offsets else None)
    report['interpretation']='Transport-frame prefix only. Peak timestamps can precede their transport frame; their future maximum delay is not established.'
    return report


def percentile(values, p):
    ordered=sorted(values)
    if not ordered: return None
    return ordered[min(len(ordered)-1, math.ceil(p*len(ordered))-1)]


def audit_progress(rows, h5_frames, sample_rate_hz, live_rows):
    result=dict(schema='mea.timing.audit.v1',raw_progress_passed=False,raw_freshness_passed=False,
                receive_clock_fit_passed=False,wall_clock_stable=False,
                live_spike_window_complete=False,absolute_hardware_clock_offset_ns=None,
                stimulation_clock_aligned=False,errors=[],alpha=1,r2_reward_enabled=False)
    errors=result['errors']
    try:
        if len(rows)<4 or rows[0].get('protocol')!='mea.raw_progress.v1' or rows[0].get('type')!='session':
            raise ValueError('progress session missing')
        if any(row.get('seq')!=i for i,row in enumerate(rows)): raise ValueError('progress sequence')
        if rows[-1].get('type')!='end': raise ValueError('progress footer missing')
        for k in ('raw_open_status','filtered_open_status'):
            if rows[0].get(k)!=0: raise ValueError('SDK open failure')
        footer=rows[-1]
        if footer.get('clean_end') is not True or footer.get('raw_contiguous') is not True or any(footer.get(k)!=0 for k in ('raw_close_status','filtered_close_status','errors')):
            raise ValueError('unclean progress end')
        started=integer(rows[0]['mono_ns'],'session clock')
        blocks=[]; total=0; prior=None; stale=[]; wall_offsets=[]; emitted=started; seen=0
        for row in rows[1:-1]:
            if row['type']=='heartbeat':
                now=integer(row['mono_ns'],'heartbeat clock'); age=integer(row['since_raw_ns'],'heartbeat age')
                seen_count=integer(row['raw_frames'],'heartbeat frames')
                previous_receive=blocks[-1]['recv_end_ns'] if blocks else started
                if now<emitted or not max(total,seen)<=seen_count<total+1000 or age>now-previous_receive:
                    raise ValueError('invalid heartbeat timing/count')
                if seen_count==0 and age!=now-started: raise ValueError('startup age mismatch')
                emitted=now;seen=seen_count;stale.append(age);continue
            if row['type']!='raw_block': raise ValueError('progress fault/unknown packet')
            start=integer(row['start_frame'],'start'); end=integer(row['end_frame'],'end')
            count=integer(row['count'],'count',1)
            if count>1000 or end-start!=count or (prior is not None and start!=prior):
                raise ValueError('raw block discontinuity')
            total+=count
            if row['cumulative_frames']!=total or total<seen: raise ValueError('raw cumulative count')
            before=integer(row['recv_begin_ns'],'receive before'); after=integer(row['recv_end_ns'],'receive after')
            wb=integer(row['wall_before_ns'],'wall before'); wa=integer(row['wall_after_ns'],'wall after')
            wall=integer(row['wall_ns'],'wall')
            previous_receive=blocks[-1]['recv_end_ns'] if blocks else started
            if after<before or before<previous_receive or wa<wb or wb<after or wa<emitted:
                raise ValueError('timestamp bracket')
            emitted=wa;seen=total
            if blocks and after<=blocks[-1]['recv_end_ns']: raise ValueError('monotonic clock regressed')
            wall_offsets.append(wall-(wb+wa)/2)
            blocks.append(row); prior=end
        if len(blocks)<3 or total!=footer['raw_frames']: raise ValueError('progress frame count')
        if not len(h5_frames) or any(int(b)!=int(a)+1 for a,b in zip(h5_frames,h5_frames[1:])):
            raise ValueError('H5 frame discontinuity')
        start,end=blocks[0]['start_frame'],blocks[-1]['end_frame']
        first_receive=integer(blocks[0]['first_recv_ns'],'first receive')
        if not started<=first_receive<=blocks[0]['recv_end_ns'] or any(b['first_recv_ns']!=first_receive for b in blocks):
            raise ValueError('first receive timestamp')
        if not start<=int(h5_frames[0])<=int(h5_frames[-1])<end: raise ValueError('H5 not covered by raw progress')
        result.update(raw_progress_passed=True,raw_interval=[start,end],raw_frames=total,progress_blocks=len(blocks),
                      h5_interval=[int(h5_frames[0]),int(h5_frames[-1])+1])
        rate=number(sample_rate_hz,'sample rate')
        if rate<=0: raise ValueError('invalid sample rate')
        # Arrival-time fit. Device acquisition time can precede these observations.
        origin_frame=blocks[0]['end_frame']-1; origin_ns=blocks[0]['recv_end_ns']
        xs=[(b['end_frame']-1-origin_frame)/rate for b in blocks]
        ys=[(b['recv_end_ns']-origin_ns)/1e9 for b in blocks]
        mx=sum(xs)/len(xs); my=sum(ys)/len(ys); xx=sum((x-mx)**2 for x in xs)
        if xx<=0: raise ValueError('zero timing span')
        slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/xx; intercept=my-slope*mx
        if slope<=0: raise ValueError('invalid receive clock fit')
        residuals=[(y-intercept-slope*x)*1000 for x,y in zip(xs,ys)]
        gaps=[(b['recv_end_ns']-a['recv_end_ns'])/1e6 for a,b in zip(blocks,blocks[1:])]
        p95=percentile([abs(v) for v in residuals],.95); ppm=(1/slope-1)*1e6
        startup=(first_receive-started)/1e6
        tail_age=0
        if 'end_mono_ns' in footer:
            end_time=integer(footer['end_mono_ns'],'end timestamp')
            if end_time<emitted:raise ValueError('end clock regressed')
            tail_age=(end_time-blocks[-1]['recv_end_ns'])/1e6
        max_age=max([startup,tail_age]+[v/1e6 for v in stale]+gaps)
        wall_range=(max(wall_offsets)-min(wall_offsets))/1e6
        result.update(apparent_rate_hz=rate/slope,apparent_rate_deviation_ppm=ppm,
                      arrival_fit_p95_abs_residual_ms=p95,arrival_fit_max_abs_residual_ms=max(map(abs,residuals)),
                      max_progress_or_heartbeat_age_ms=max_age,wall_offset_range_ms=wall_range,
                      startup_first_frame_wait_ms=startup,
                      terminal_frame_age_ms=tail_age,
                      raw_freshness_passed=max_age<=250,wall_clock_stable=wall_range<=50,
                      receive_clock_fit_passed=abs(ppm)<=1000 and p95<=20 and wall_range<=50,
                      thresholds=dict(progress_ms=250,arrival_fit_p95_ms=20,apparent_rate_ppm=1000,wall_offset_range_ms=50),
                      fit=dict(origin_frame=origin_frame,origin_mono_raw_ns=origin_ns,slope=slope,intercept_seconds=intercept))
        result['blocks_without_new_filtered_events']=sum(b.get('filtered_events')==a.get('filtered_events')
            for a,b in zip(blocks,blocks[1:]) if 'filtered_events' in a and 'filtered_events' in b)
        relative=[]
        for row in live_rows:
            if row.get('type')!='events': continue
            recv=integer(row['recv_end_ns'],'event receive')
            for frame,channel,well,amp in row['events']:
                if start<=frame<end:
                    predicted=(intercept+slope*((frame-origin_frame)/rate))*1e9
                    relative.append(((recv-origin_ns)-predicted)/1e6)
        result['filtered_relative_arrival_ms']=dict(count=len(relative),min=min(relative) if relative else None,
             p50=percentile(relative,.5),p95=percentile(relative,.95),max=max(relative) if relative else None)
        result['interpretation']='Receive-clock fit and relative stream arrival only; unknown fixed latency is not removed.'
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        errors.append(str(exc)); result['raw_progress_passed']=False
    return result


def clock_bounds(samples):
    if not samples: raise ValueError('no clock probes')
    bounds=[]; rtts=[]; wall_errors=[]; between=[]; previous=None
    for row in samples:
        c0,s1,s2,c3=[integer(row[k],k) for k in ('c0','s1','s2','c3')]
        cm0,cm3,sm1,sm2=[integer(row[k],k) for k in ('cm0','cm3','sm1','sm2')]
        if c3<c0 or s2<s1: raise ValueError('clock regressed')
        if cm3<cm0 or sm2<sm1: raise ValueError('monotonic clock regressed')
        if row['client_elapsed_ns']!=cm3-cm0 or row['server_elapsed_ns']!=sm2-sm1: raise ValueError('elapsed clock mismatch')
        if previous:
            if cm0<previous['cm3'] or sm1<previous['sm2']: raise ValueError('probe clock order')
            between.extend([abs((c0-previous['c0'])-(cm0-previous['cm0'])),
                            abs((s1-previous['s1'])-(sm1-previous['sm1']))])
        rtt=(c3-c0)-(s2-s1)
        if rtt<0: raise ValueError('impossible exchange duration')
        lo,hi=s2-c3,s1-c0
        bounds.append((lo,hi)); rtts.append(rtt)
        wall_errors.extend([abs((c3-c0)-row['client_elapsed_ns']),abs((s2-s1)-row['server_elapsed_ns'])])
        previous=row
    if max(wall_errors+between)>5_000_000: raise ValueError('wall clock changed within/between probes')
    # No stationarity assumption: retain the last exchange's interval, not an
    # intersection that may exclude an offset after clocks move between probes.
    lo,hi=bounds[-1];best=min(range(len(rtts)),key=rtts.__getitem__)
    return dict(offset_definition='source wall time minus deployment wall time',offset_bounds_ns=[lo,hi],
                offset_bounds_apply_to='last exchange only; no extrapolation',constant_offset_assumed=False,
                best_sample_offset_bounds_ns=list(bounds[best]),best_sample_index=best,
                midpoint_estimate_ns=(lo+hi)/2,best_rtt_ns=min(rtts),p95_rtt_ns=percentile(rtts,.95),
                probes=len(samples),path_symmetry_proven=False,absolute_utc_accuracy_proven=False,
                maximum_wall_vs_monotonic_difference_ns=max(wall_errors),
                maximum_between_probe_wall_change_ns=max([0]+between))
