"""Finalized, audited recording windows. No hardware or stimulus interface."""
from dataclasses import asdict,dataclass,field
from pathlib import Path
from .mea_protocol import ResearchStream,canonical,digest,identifier,integer,number,strict_json
from .mea_recording import sha256
from .mea_snapshot import FEATURE_NAMES,Normalization,vector

MANIFEST_SHA256='2a85126aac33f824b2523ea9459e16d1264317a7453d9a653e739812c37228c1'
RECORDING_CONTRACT='mea_rl_recording_v1'


@dataclass(frozen=True)
class RecordingSnapshot:
    recording_sha256: str
    session_id: str
    context_id: str
    channel_map_id: str
    index: int
    well_id: int
    mapped_channels: int
    sample_rate_hz: float
    window_start_frame: int
    window_end_frame: int
    window_spikes_digest: str
    feature_values: tuple
    normalization: Normalization
    schema: str=field(default='mea.recorded_snapshot.v1',init=False)
    source: str=field(default='recording',init=False)
    association: str=field(default='sequential_offline',init=False)
    stimulus_event_id: None=field(default=None,init=False)
    coverage_basis: str=field(default='audited_finalized_h5',init=False)
    normalized_values: tuple=field(init=False)
    normalization_id: str=field(init=False)
    snapshot_id: str=field(init=False)

    def __post_init__(self):
        for key in ('recording_sha256','session_id','context_id','channel_map_id','window_spikes_digest'):
            identifier(getattr(self,key),key)
        for key in ('index','well_id','window_start_frame','window_end_frame'):integer(getattr(self,key),key)
        integer(self.mapped_channels,'mapped channels',1)
        rate=number(self.sample_rate_hz,'rate');duration=self.window_end_frame-self.window_start_frame
        if rate<=0 or duration<=0 or not isinstance(self.normalization,Normalization):raise ValueError('invalid window/normalization')
        values=vector(self.feature_values,'features');count,freq,fraction,amplitude=values
        if (any(x<0 for x in values) or count!=int(count) or fraction>1 or
            freq!=count*rate/(duration*self.mapped_channels) or
            (count==0 and values!=(0.,0.,0.,0.)) or (count>0 and fraction<=0)):
            raise ValueError('inconsistent recorded features')
        object.__setattr__(self,'feature_values',values)
        object.__setattr__(self,'normalized_values',self.normalization.transform(values))
        object.__setattr__(self,'normalization_id',self.normalization.normalization_id)
        object.__setattr__(self,'snapshot_id',digest(self._payload()))

    def _payload(self):
        return {k:getattr(self,k) for k in self.__dataclass_fields__ if k not in ('normalization','snapshot_id')}|{'normalization':asdict(self.normalization)}

    def to_dict(self):return strict_json(canonical(self._payload()|{'snapshot_id':self.snapshot_id}))

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!=set(cls.__dataclass_fields__):raise ValueError('snapshot schema')
        try:
            kwargs={k:v for k,v in data.items() if cls.__dataclass_fields__[k].init}
            kwargs['normalization']=Normalization(**kwargs['normalization']);result=cls(**kwargs)
            if result.to_dict()!=data:raise ValueError('snapshot identity/derived data differs')
            return result
        except (KeyError,TypeError,OverflowError) as exc:raise ValueError('invalid recording snapshot') from exc


class AuditedRecording:
    def __init__(self,package,*,manifest_sha256=MANIFEST_SHA256,window_seconds=1):
        self.root=Path(package).resolve()
        if sha256(self.root/'manifest.json')!=manifest_sha256:raise ValueError('sealed manifest changed')
        manifest=strict_json((self.root/'manifest.json').read_text(encoding='utf-8'))
        name=manifest['accepted_run']
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):raise ValueError('invalid accepted run path')
        run=(self.root/name).resolve()
        if not run.is_relative_to(self.root):raise ValueError('run escaped package')
        self.input_hashes={}
        for n in ('baseline.raw.h5','events.jsonl','acquisition.json','audit.json','recording.jsonl'):
            path=(run/n).resolve();expected=manifest['files'][name+'/'+n]
            if not path.is_relative_to(self.root) or path.stat().st_size!=expected['size'] or sha256(path)!=expected['sha256']:
                raise ValueError('audited input changed: '+n)
            self.input_hashes[n]=expected['sha256']
        self.manifest_sha256=manifest_sha256
        audit=strict_json((run/'audit.json').read_text(encoding='utf-8'))
        acquisition=strict_json((run/'acquisition.json').read_text(encoding='utf-8'))
        if (audit['offline_integrity_passed'] is not True or audit['errors'] or acquisition['success'] is not True or
            acquisition['stimulation'] is not False or acquisition['alpha']!=1 or acquisition['r2_reward_enabled'] is not False or
            str(audit['metadata']['chip_id'])!='22021' or str(acquisition['expected_chip_id'])!='22021'):
            raise ValueError('unaudited/wrong-chip/stimulated source')
        if any(audit['inputs'][n]!=self.input_hashes[n] for n in ('baseline.raw.h5','events.jsonl','acquisition.json')):
            raise ValueError('audit not bound to source files')
        stream=ResearchStream(max_spikes=1000000)
        with (run/'recording.jsonl').open('rb') as f:
            for b in iter(lambda:f.read(65536),b''):stream.feed(b)
        stream.finish();session=stream.session
        if (session is None or session.source!='recording' or stream.stimuli or session.well_id!=0 or
            [session.start_frame,stream.watermark]!=audit['audit_interval'] or len(stream.spikes)!=audit['matched_events'] or
            session.channel_map_id!=audit['channel_map_id'] or digest(session.channels)!=session.channel_map_id or
            session.sample_rate_hz!=audit['metadata']['sample_rate_hz'] or
            [list(p) for p in session.channels]!=audit['metadata']['mapping'] or audit['metadata']['well_id']!=0):
            raise ValueError('recording protocol/audit identity mismatch')
        if any(not session.start_frame<=e[0]<stream.watermark for e in stream.spikes):raise ValueError('event outside audited interval')
        span=number(window_seconds,'window seconds')*session.sample_rate_hz
        if span<=0 or span!=int(span):raise ValueError('positive whole-frame windows required')
        self.window_frames=int(span);self.session=session;self.end_frame=stream.watermark
        self._spikes=tuple(stream.spikes)
        self.window_count=(self.end_frame-session.start_frame)//self.window_frames
        if self.window_count<1:raise ValueError('no complete recording windows')

    def snapshot(self,index,context_id,normalization=None):
        integer(index,'window index')
        if index>=self.window_count:raise ValueError('recording exhausted; no wraparound')
        start=self.session.start_frame+index*self.window_frames;end=start+self.window_frames
        events=tuple(e for e in self._spikes if start<=e[0]<end);channels=len(self.session.channels);count=len(events)
        values=(float(count),count*self.session.sample_rate_hz/(self.window_frames*channels),
                len({e[1] for e in events})/channels,sum(abs(e[2]) for e in events)/count if count else 0.)
        return RecordingSnapshot(self.input_hashes['recording.jsonl'],self.session.session_id,context_id,
            self.session.channel_map_id,index,self.session.well_id,channels,self.session.sample_rate_hz,start,end,
            digest(events),values,normalization or Normalization.identity())

    def summary(self):
        end=self.session.start_frame+self.window_count*self.window_frames
        return dict(source='recording',chip_id='22021',session_id=self.session.session_id,
            manifest_sha256=self.manifest_sha256,inputs=self.input_hashes,window_frames=self.window_frames,
            windows=self.window_count,audit_start_frame=self.session.start_frame,audit_end_frame=self.end_frame,
            complete_windows_end_frame=end,tail_frames=self.end_frame-end,
            total_events=len(self._spikes),tail_events=sum(e[0]>=end for e in self._spikes),
            stimulation=False,association='sequential_offline',normalization='identity_engineering_only')
