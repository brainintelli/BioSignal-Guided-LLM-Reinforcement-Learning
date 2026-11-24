"""Analyze a closed baseline H5 offline; never enables R2 or accesses a device."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from agent.mea_baseline import analyze_arrays
from agent.mea_recording import load_h5, sha256


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_h5_identity_types(source):
    """Reject fractional identities before the legacy loader performs int casts."""
    with h5py.File(source, 'r') as f:
        stores = list(f['data_store'])
        if len(stores) != 1:
            raise ValueError('one data store required')
        store = f['data_store'][stores[0]]
        groups = list(store['groups'])
        if len(groups) != 1:
            raise ValueError('one raw group required')
        raw = store['groups'][groups[0]]
        for dataset in (store['well_id'], store['start_time'], store['stop_time'],
                        raw['channels'], raw['frame_nos']):
            if dataset.dtype.kind not in 'iu':
                raise ValueError('integer H5 identity dtype required: '+dataset.name)
        for dataset, fields in ((store['settings/mapping'], ('channel', 'electrode')),
                                (store['spikes'], ('frameno', 'channel'))):
            for field in fields:
                if not dataset.dtype.fields or field not in dataset.dtype.fields or dataset.dtype.fields[field][0].kind not in 'iu':
                    raise ValueError('integer H5 identity field required: '+dataset.name+'/'+field)
        for dataset in (store['well_id'], store['start_time'], store['stop_time']):
            if dataset.size != 1 or int(dataset[()].reshape(-1)[0]) < 0:
                raise ValueError('one nonnegative scalar identity required: '+dataset.name)


def sample_raw(source, metadata):
    """Twelve fixed spans, never an assertion about every raw voltage sample."""
    with h5py.File(source, 'r') as f:
        store = f['data_store'][list(f['data_store'])[0]]
        group = store['groups'][metadata['raw_group']]
        raw = group['raw']
        count = max(1, int(metadata['sample_rate_hz'] * .25))
        count = min(count, raw.shape[1])
        starts = np.unique(np.linspace(0, raw.shape[1]-count, 12, dtype=np.int64))
        low = np.full(raw.shape[0], np.inf); high = np.full(raw.shape[0], -np.inf)
        means = []; stds = []; spans = []; storage_extremes = 0; total = 0
        bounds = np.iinfo(raw.dtype) if raw.dtype.kind in 'iu' else None
        for start in starts:
            block = raw[:, start:start+count]
            low = np.minimum(low, block.min(axis=1)); high = np.maximum(high, block.max(axis=1))
            means.append(block.mean(axis=1)); stds.append(block.std(axis=1))
            if bounds:
                storage_extremes += int(np.sum((block == bounds.min) | (block == bounds.max)))
            total += block.size
            spans.append(dict(start_index=int(start), sample_count=count,
                              start_frame=int(group['frame_nos'][start]),
                              end_frame=int(group['frame_nos'][start+count-1])+1))
        channels = metadata['raw_channels']
        return dict(scope='12 evenly spaced 0.25-second spans, all routed channels; sampled only',
                    spans=spans, sampled_voltage_values=total,
                    sampled_time_per_channel_seconds=len(starts)*count/metadata['sample_rate_hz'],
                    stored_dtype=str(raw.dtype), observed_min_code=int(low.min()), observed_max_code=int(high.max()),
                    storage_dtype_extreme_values=storage_extremes,
                    adc_effective_rails_known=False, hardware_saturation_assessed=False,
                    sampled_constant_channel_count=int(np.sum(low == high)),
                    sampled_constant_channels=[channels[i] for i in np.flatnonzero(low == high)],
                    median_per_channel_span_std_uv=float(np.median(stds)*metadata['lsb_volts']*1e6),
                    median_per_channel_span_mean_code=float(np.median(means)),
                    h5_event_rows=len(store['events']) if 'events' in store else None,
                    event_log_absence_proves_no_stimulation=False)


def make_plots(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    windows = result['windows']; candidates = result['candidates']; profile = candidates['profile']
    if not windows:
        return
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    times = [w['time_seconds']+w['duration_seconds']/2 for w in windows]
    for ax, key, label, color in zip(axes[:3],
            ('rate_hz_per_channel', 'active_channel_fraction', 'coincident_event_fraction'),
            ('Events / channel / s', 'Active channel fraction', '10 ms co-occurrence fraction'),
            ('#176b87', '#3b8269', '#7955a2')):
        ax.plot(times, [w[key] for w in windows], color=color, linewidth=1.6)
        ax.set_ylabel(label); ax.grid(alpha=.18)
    if profile:
        fit = profile['split']['fit']; gap = profile['split']['embargo']; val = profile['split']['validation']
        edge = lambda i: windows[i]['time_seconds'] if i < len(windows) else windows[-1]['time_seconds']+windows[-1]['duration_seconds']
        for ax in axes:
            ax.axvspan(edge(fit[0]), edge(fit[1]), color='#54a6c2', alpha=.09)
            ax.axvspan(edge(gap[0]), edge(gap[1]), color='#888888', alpha=.15)
            ax.axvspan(edge(val[0]), edge(val[1]), color='#a687c1', alpha=.08)
        s = candidates['scores']; st = [times[v['window_index']] for v in s]
        axes[3].plot(st, [v['baseline_proximity'] for v in s], label='Baseline proximity', color='#176b87')
        axes[3].plot(st, [v['novelty'] for v in s], label='Past-only novelty', color='#b66d17')
        axes[3].legend(loc='upper left', frameon=False); axes[3].set_ylim(-.04, 1.04)
    axes[3].set_ylabel('Exploratory score'); axes[3].set_xlabel('Seconds after recording start')
    axes[3].grid(alpha=.18)
    scope = 'Mapped-channel subset; source QC failed' if 'source_qc' in result else 'Unstimulated baseline'
    fig.suptitle('Chip 22554 | '+scope+' | 5 s windows\nFit (blue) / 10 s gap (gray) / validation (purple). Scores are not training rewards.', fontsize=12)
    fig.savefig(output/'baseline_candidates.png', dpi=160)
    fig.savefig(output/'baseline_candidates.svg')
    plt.close(fig)


def routed_scope_analysis(metadata, frames, spikes, original):
    """Explicit supplementary scope; preserve the rejected full-source verdict."""
    allowed = {'events_outside_recording', 'unmapped_events'}
    if not original['qc']['errors'] or not set(original['qc']['errors']) <= allowed:
        raise ValueError('subset requires only independently audited scope errors')
    channels = {p[0] for p in metadata['mapping']}; lo, hi = int(frames[0]), int(frames[-1])+1
    indices = [i for i, s in enumerate(spikes) if s[1] not in channels or not lo <= s[0] < hi]
    excluded = set(indices); retained = [s for i, s in enumerate(spikes) if i not in excluded]
    result = analyze_arrays(metadata, frames, retained)
    if not result['qc']['structural_passed']:
        raise ValueError('routed subset still fails structural QC')
    result['source_qc'] = original['qc']
    result['analysis_scope'] = 'supplementary mapped-channel raw-time subset; full-source QC remains failed'
    result['scope_exclusions'] = dict(count=len(indices), event_indices=indices,
                                      rules=['channel absent from recorded mapping', 'frame outside raw interval'])
    if result['candidates']['profile'] is not None:
        result['candidates']['status'] = 'exploratory_subset_only'
        result['candidates']['profile'].update(source_full_structural_passed=False,
                                               scope=result['analysis_scope'],
                                               source_excluded_events=len(indices))
    return result, retained


def analyze_file(source, output, *, expected_sha256, plots=True, routed_subset=False):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    before = source.stat(); actual = sha256(source)
    if actual != expected_sha256:
        raise ValueError('source SHA-256 does not match frozen source identity')
    validate_h5_identity_types(source)
    metadata, frames, spikes = load_h5(source)
    result = analyze_arrays(metadata, frames, spikes)
    if routed_subset:
        result, spikes = routed_scope_analysis(metadata, frames, spikes, result)
    result['metadata'] = metadata
    result['raw_sample_qc'] = sample_raw(source, metadata)
    result['source'] = dict(path=str(source), sha256=actual, size_bytes=before.st_size)
    result['generated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['stimulation_status'] = 'unstimulated_as_reported_by_user; H5 events not a complete stimulation audit'
    result['sensitivity'] = {}
    for width in (1, 10):
        sensitivity = analyze_arrays(metadata, frames, spikes, window_seconds=width)
        windows = sensitivity['windows']
        result['sensitivity'][str(width)] = dict(window_seconds=width, windows=len(windows),
            zero_event_windows=sum(w['spike_count'] == 0 for w in windows),
            candidate_status=sensitivity['candidates']['status'],
            validation=sensitivity['candidates'].get('validation'),
            interpretation='prespecified descriptive sensitivity; not candidate selection')
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(source) != actual:
        raise ValueError('source changed during analysis')
    result['source']['unchanged_during_analysis'] = True
    root = Path(__file__).resolve().parents[1]
    result['implementation'] = {str(p.relative_to(root)): sha256(p) for p in
        (root/'agent/mea_baseline.py', root/'agent/mea_recording.py', Path(__file__).resolve())}
    result['runtime'] = dict(python=sys.version, numpy=np.__version__, h5py=h5py.__version__)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'analysis.json', result)
    profile = result['candidates']['profile']
    write_json(output/'candidate_profile.json', dict(
        source_sha256=actual, chip_id=metadata['chip_id'], r2_training_enabled=False,
        source_full_structural_passed=result.get('source_qc', result['qc'])['structural_passed'],
        scope=result.get('analysis_scope', 'full source'),
        status=result['candidates']['status'], profile=profile,
        qualification='exploratory within-session baseline scale only; no efficacy or cross-session calibration'))
    if result['windows']:
        with (output/'windows.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(result['windows'][0])); writer.writeheader(); writer.writerows(result['windows'])
    if plots:
        make_plots(result, output)
    write_json(output/'manifest.json', {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--routed-subset', action='store_true', help='Supplementary scope only; keeps failed full-source verdict')
    args = parser.parse_args()
    result = analyze_file(args.input, args.output, expected_sha256=args.expected_sha256, routed_subset=args.routed_subset)
    print(json.dumps(dict(qc=result['qc'], source_qc=result.get('source_qc'), candidate_status=result['candidates']['status'],
                         validation=result['candidates'].get('validation')), ensure_ascii=False, indent=2))
    return 0 if result['qc']['structural_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
