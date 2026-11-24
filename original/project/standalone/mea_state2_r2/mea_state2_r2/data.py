"""Admitted 28080 event replay with a frozen training-baseline transform."""
from agent.mea_stim_offline import StimRecording
from .features import SCHEMA, NAMES, digest, extract, fit, transform, validate_snapshot
from .protocol import PIN, assignment


class Dataset:
    def __init__(self, source):
        self.recordings = {phase: StimRecording(source, pin=PIN, phase=phase) for phase in ('before', 'after')}
        self.features = {}
        for phase, data in self.recordings.items():
            channels = tuple(c for c, _ in data.mapping)
            self.features[phase] = [extract(row['events'], row['start'], row['end'], channels, 20000) for row in data.rows]
        self.normalization = fit(self.features['before'][:385])
        self.normalization.update(source_pin=PIN, fit_phase='before', fit_range=[0, 385])
        self.normalization_id = digest(self.normalization)

    def snapshot(self, position, phase, context_id, arm, block, split):
        order = assignment(block, arm, split)
        if type(position) is not int or not 0 <= position < len(order) or phase not in self.recordings:
            raise ValueError('Outside finite registered windows')
        if split == 'train' and phase != 'after':
            raise ValueError('Training must use after windows')
        index = order[position]
        data = self.recordings[phase]
        row = data.rows[index]
        values = self.features[phase][index]
        payload = dict(schema=SCHEMA, names=list(NAMES), source='recording', association='historical_no_game_causality',
            source_pin=PIN, phase=phase, window_index=index, window_start=row['start'], window_end=row['end'],
            channel_map_id=data.map_id, events_sha256=digest(row['events']), context_id=context_id,
            features=list(values), normalized=list(transform(values, self.normalization)), normalization_id=self.normalization_id)
        payload['snapshot_id'] = digest(payload)
        validate_snapshot(payload, context_id, self.normalization_id)
        return payload

    def summary(self):
        return dict(source_pin=PIN, phases={k: v.summary() for k, v in self.recordings.items()},
                    normalization=self.normalization, normalization_id=self.normalization_id,
                    schema=SCHEMA, names=list(NAMES), historical_game_causality=False)
