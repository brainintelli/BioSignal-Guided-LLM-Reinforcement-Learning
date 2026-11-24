"""Read-only verification of this captured artifact using the P0 CSV decoder."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[2]))
from agent.mea_protocol import LegacyCsvDecoder


def verify():
    metadata = json.loads((DIRECTORY / 'capture.json').read_text(encoding='utf-8'))
    raw = (DIRECTORY / 'stream.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['raw_sha256']:
        raise ValueError('Raw file hash mismatch')
    decoder = LegacyCsvDecoder(time_unit='ms')
    spikes = []
    for index in range(0, len(raw), 37):
        spikes.extend(decoder.feed(raw[index:index + 37]))
    decoder.finish()
    if not spikes:
        raise ValueError('No received events; cannot infer silence')
    offset = 0
    chunks = [json.loads(line) for line in (DIRECTORY / 'receive_chunks.jsonl').read_text().splitlines()]
    for chunk in chunks:
        if chunk['offset'] != offset or chunk['bytes'] <= 0:
            raise ValueError('Broken local receive byte index')
        offset += chunk['bytes']
    if offset != len(raw) or offset != metadata['received_bytes'] or len(chunks) != metadata['chunks']:
        raise ValueError('Receive totals mismatch')
    timestamps = [s.host_time_ms for s in spikes]
    counts = Counter(s.channel for s in spikes)
    saved = json.loads((DIRECTORY / 'quality.json').read_text())
    if len(spikes) != saved['csv_records'] or len(counts) != saved['observed_channel_count']:
        raise ValueError('Quality summary count mismatch')
    regressions = sum(b < a for a, b in zip(timestamps, timestamps[1:]))
    if regressions != saved['host_timestamp_regressions']:
        raise ValueError('Timestamp audit mismatch')
    if metadata['research_qualified'] is not False or metadata['alpha'] != 1 or metadata['r2_enabled'] is not False:
        raise ValueError('Capture scope metadata mismatch')
    return {'passed': True, 'csv_records': len(spikes), 'observed_channels': len(counts),
            'raw_sha256': metadata['raw_sha256'], 'local_receive_index_verified': True,
            'host_timestamp_regressions': regressions, 'research_qualified': False,
            'source_loss_proven_absent': False, 'device_frame_coverage_verified': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
