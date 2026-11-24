"""Opt-in offline H5-metadata subset; never claims independent hardware readback."""
from pathlib import Path
from .mea_protocol import ResearchStream,digest,number,strict_json
from .mea_recording import sha256,recorded_h5_selection
from .mea_replay import AuditedRecording

SCHEMA='mea.finalized_h5_subset_admission.v1'
PACKAGE_SCHEMA='mea.sealed.h5_subset.v1'


def resolve_accepted_run(package,manifest):
    root=Path(package).resolve();name=manifest['accepted_run']
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):raise ValueError('run name')
    run=(root/name).resolve()
    if not run.is_relative_to(root) or not run.is_dir():raise ValueError('invalid accepted run directory')
    return run


def validate_confirmation(acquisition,metadata,confirmation,expected_chip_id):
    if (acquisition.get('success') is not True or acquisition.get('errors') or
        acquisition.get('stimulation') is not False or acquisition.get('alpha')!=1 or
        acquisition.get('r2_reward_enabled') is not False):raise ValueError('capture was not clean/unstimulated/R1-only')
    if (confirmation.get('schema')!='mea.user_chip_confirmation.v1' or
        confirmation.get('source')!='explicit_user_reply' or
        not isinstance(confirmation.get('statement'),str) or not confirmation['statement'].strip() or
        confirmation.get('original_expected_chip_id')!=str(acquisition['expected_chip_id']) or
        confirmation.get('chip_id')!=expected_chip_id or str(metadata['chip_id'])!=expected_chip_id):
        raise ValueError('explicit chip confirmation differs from source or requested identity')


class H5MappedRecording(AuditedRecording):
    def __init__(self,package,*,manifest_sha256,expected_chip_id,window_seconds=1):
        self.root=Path(package).resolve()
        if sha256(self.root/'manifest.json')!=manifest_sha256:raise ValueError('sealed manifest changed')
        manifest=strict_json((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema')!=PACKAGE_SCHEMA:raise ValueError('wrong subset package schema')
        run=resolve_accepted_run(self.root,manifest);name=run.name
        names=('baseline.raw.h5','events.jsonl','acquisition.json','audit.json','chip_confirmation.json','admission.json','recording.jsonl')
        self.input_hashes={}
        for n in names:
            p=(run/n).resolve();item=manifest['files'][name+'/'+n]
            if not p.is_relative_to(self.root) or p.stat().st_size!=item['size'] or sha256(p)!=item['sha256']:raise ValueError('source changed: '+n)
            self.input_hashes[n]=item['sha256']
        admission=strict_json((run/'admission.json').read_text(encoding='utf-8'));meta=admission['metadata']
        acquisition=strict_json((run/'acquisition.json').read_text(encoding='utf-8'));confirmation=strict_json((run/'chip_confirmation.json').read_text(encoding='utf-8'))
        validate_confirmation(acquisition,meta,confirmation,expected_chip_id)
        if (admission.get('schema')!=SCHEMA or admission.get('offline_integrity_passed') is not True or admission['errors'] or
            admission.get('mapping_policy')!='finalized_h5_recorded_subset' or
            admission.get('mapping_basis')!='finalized_h5_metadata' or admission.get('hardware_mapping_independently_verified') is not False):
            raise ValueError('subset admission failed or basis changed')
        if any(admission['inputs'][n]!=self.input_hashes[n] for n in names[:5]):raise ValueError('admission not bound to original inputs')
        selected=recorded_h5_selection(meta)
        if selected!=admission['analysis_mapping'] or digest(selected)!=admission['channel_map_id']:raise ValueError('analysis subset changed')
        stream=ResearchStream(max_spikes=1000000)
        with (run/'recording.jsonl').open('rb') as f:
            for b in iter(lambda:f.read(65536),b''):stream.feed(b)
        stream.finish();session=stream.session
        if (session is None or session.source!='recording' or stream.stimuli or session.well_id!=0 or meta['well_id']!=0 or
            [session.start_frame,stream.watermark]!=admission['audit_interval'] or len(stream.spikes)!=admission['matched_events'] or
            session.channel_map_id!=admission['channel_map_id'] or digest(session.channels)!=session.channel_map_id or
            session.sample_rate_hz!=meta['sample_rate_hz'] or [list(p) for p in session.channels]!=selected):
            raise ValueError('export not bound to admitted subset')
        if any(not session.start_frame<=e[0]<stream.watermark for e in stream.spikes):raise ValueError('event outside audited interval')
        span=number(window_seconds,'window seconds')*session.sample_rate_hz
        if span<=0 or span!=int(span):raise ValueError('positive whole-frame windows required')
        self.manifest_sha256=manifest_sha256;self.chip_id=expected_chip_id;self.window_frames=int(span)
        self.session=session;self.end_frame=stream.watermark;self._spikes=tuple(stream.spikes)
        self.window_count=(self.end_frame-session.start_frame)//self.window_frames
        if self.window_count<1:raise ValueError('no complete recording windows')

    def summary(self):
        return super().summary()|dict(chip_id=self.chip_id,mapping_basis='finalized_h5_metadata',
            hardware_mapping_independently_verified=False,analysis_channels=len(self.session.channels),admission_schema=SCHEMA)
