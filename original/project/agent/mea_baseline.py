"""Offline baseline diagnostics and exploratory scores, never training rewards."""
import math
import numpy as np


FEATURES = ('log1p_rate_hz_per_channel', 'active_channel_fraction', 'coincident_event_fraction')


def _summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                minimum=float(a.min()), maximum=float(a.max()))


def calibrate_candidates(windows, window_seconds=5):
    """Fit the first contiguous half; score only later windows using past history."""
    n = len(windows); fit_end = n // 2
    validation_start = fit_end + math.ceil(10 / window_seconds)
    result = dict(status='insufficient_temporal_windows', profile=None, scores=[],
                  response_repeatability=dict(status='not_estimable_without_repeated_stimuli'))
    if fit_end < 8 or n - validation_start < 8:
        return result
    x = np.array([[math.log1p(w['rate_hz_per_channel']), w['active_channel_fraction'],
                   w['coincident_event_fraction']] for w in windows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('finite window features required')
    fit = x[:fit_end]
    center = np.median(fit, axis=0)
    scale = 1.4826 * np.median(np.abs(fit - center), axis=0)
    active = scale > 1e-12
    if not active.any():
        result['status'] = 'unidentifiable_constant_features'
        return result
    z = (x[:, active] - center[active]) / scale[active]
    # Every reference index precedes the scored index. Fitting these scale
    # parameters is offline; only validation scores represent prospective use.
    distances = [float(np.sqrt(np.mean((z[:i] - z[i]) ** 2, axis=1)).min())
                 for i in range(1, fit_end)]
    novelty_scale = float(np.quantile(distances, .95))
    profile = dict(schema='mea.r2.baseline_candidate.v1', exploratory_only=True,
                   r2_training_enabled=False, alpha=1, feature_names=list(FEATURES),
                   window_seconds=window_seconds, cooccurrence_bin_seconds=.01,
                   center=center.tolist(), scale=scale.tolist(), active_features=active.tolist(),
                   excluded_zero_scale_features=[FEATURES[i] for i in np.flatnonzero(~active)],
                   split=dict(fit=[0, fit_end], embargo=[fit_end, validation_start],
                              validation=[validation_start, n]),
                   minimum_windows_per_segment=8,
                   distance='root_mean_square_standardized_active_features',
                   baseline_proximity='exp(-0.5 * distance_to_fitted_center ** 2)',
                   novelty='min(nearest_past_distance / fit_past_distance_p95, 1)',
                   novelty_distance_p95=novelty_scale,
                   history_rule='all prior windows, append after scoring; embargo history is past-only',
                   direction_status='engineering conventions; biological utility unvalidated',
                   missing_requirements=['cross_session_retest', 'stimulus_condition_calibration',
                                         'artifact_controls', 'task_input_controls',
                                         'reward_timing_and_alpha_registration', 'on_policy_comparison'])
    scores = []
    for i in range(validation_start, n):
        distance = float(np.sqrt(np.mean(z[i] ** 2)))
        nearest = float(np.sqrt(np.mean((z[:i] - z[i]) ** 2, axis=1)).min())
        scores.append(dict(window_index=i, history_last_window=i-1,
                           baseline_distance=distance, baseline_proximity=math.exp(-.5 * distance ** 2),
                           nearest_past_distance=nearest,
                           novelty=min(nearest / novelty_scale, 1.) if novelty_scale > 1e-12 else None))
    result.update(status='exploratory_baseline_only', profile=profile, scores=scores,
                  validation=dict(
                      baseline_proximity=_summary([s['baseline_proximity'] for s in scores]),
                      novelty=_summary([s['novelty'] for s in scores if s['novelty'] is not None]),
                      novelty_scale_identifiable=novelty_scale > 1e-12,
                      novelty_clipped_fraction=float(np.mean([s['novelty'] == 1 for s in scores]))
                      if novelty_scale > 1e-12 else None,
                      median_shift_in_fit_mad_units=np.median(z[validation_start:], axis=0).tolist(),
                      inactive_feature_changed_fraction={FEATURES[i]: float(np.mean(
                          np.abs(x[validation_start:, i] - center[i]) > 1e-12)) for i in np.flatnonzero(~active)},
                      inactive_feature_fit_changed_fraction={FEATURES[i]: float(np.mean(
                          np.abs(fit[:, i] - center[i]) > 1e-12)) for i in np.flatnonzero(~active)},
                      fraction_any_feature_over_3_fit_mad=float(np.mean(
                          np.any(np.abs(z[validation_start:]) > 3, axis=1)))))
    return result


def analyze_arrays(metadata, frames, spikes, window_seconds=5, guard_seconds=1):
    """Check the stored recording and describe full windows; no live claims."""
    rate = float(metadata['sample_rate_hz']); lsb = float(metadata['lsb_volts'])
    if not all(math.isfinite(v) and v > 0 for v in (rate, lsb, window_seconds)) or not math.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError('invalid recording/window scale')
    if not rate.is_integer():
        raise ValueError('integer device sampling rate required for exact temporal bins')
    errors = []
    qc = dict(errors=errors, structural_passed=False, online_coverage_proven=False,
              paired_event_stream_verified=False, expected_chip_id='22554',
              spike_detection_sensitivity_verified=False, adc_saturation_verified=False)
    result = dict(schema='mea.baseline.analysis.v1', qc=qc, windows=[],
                  candidates=dict(status='qc_failed', profile=None, scores=[]))
    if str(metadata['chip_id']) != '22554':
        errors.append('chip_id_mismatch')
    frames = np.asarray(frames)
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu':
        errors.append('invalid_raw_frames'); return result
    if np.any(frames < 0) or np.any(frames > np.iinfo(np.int64).max):
        errors.append('invalid_raw_frames'); return result
    frames = frames.astype(np.int64)
    discontinuities = np.flatnonzero(np.diff(frames) != 1)
    start, end = int(frames[0]), int(frames[-1]) + 1
    duration = len(frames) / rate
    qc.update(raw_frames=len(frames), raw_interval=[start, end], raw_duration_seconds=duration,
              raw_frame_discontinuities=len(discontinuities),
              raw_discontinuity_examples=discontinuities[:10].tolist())
    if len(discontinuities):
        errors.append('raw_frame_discontinuity')
    mapping = metadata['mapping']
    valid_mapping = (isinstance(mapping, list) and bool(mapping) and
                     all(isinstance(p, list) and len(p) == 2 and
                         all(type(v) is int and v >= 0 for v in p) for p in mapping))
    if not valid_mapping:
        errors.append('invalid_mapping'); return result
    channels = [p[0] for p in mapping]
    if len(set(channels)) != len(channels) or len({p[1] for p in mapping}) != len(mapping):
        errors.append('mapping_not_one_to_one')
    raw_channels = metadata['raw_channels']
    if len(set(raw_channels)) != len(raw_channels) or set(raw_channels) != set(channels):
        errors.append('raw_channel_coverage')
    if metadata['raw_shape'] != [len(raw_channels), len(frames)]:
        errors.append('raw_shape_mismatch')
    wall_duration = (metadata['stop_time_unix_ms'] - metadata['start_time_unix_ms']) / 1000
    qc['h5_wall_duration_seconds'] = wall_duration
    if not math.isfinite(wall_duration) or abs(wall_duration - duration) > .5:
        errors.append('h5_time_duration_mismatch')
    if isinstance(spikes, np.ndarray) and spikes.dtype.names:
        events = np.column_stack([spikes['frameno'], spikes['channel'], spikes['amplitude']])
    else:
        events = np.asarray(spikes, dtype=float)
    if not events.size:
        events = np.empty((0, 3))
    if events.ndim != 2 or events.shape[1] != 3:
        errors.append('invalid_event_shape'); return result
    if not np.isfinite(events[:, :2]).all() or np.any(events[:, :2] != np.floor(events[:, :2])) or np.any(np.abs(events[:, :2]) > 2**53):
        errors.append('invalid_event_identity'); return result
    ef = events[:, 0].astype(np.int64); ec = events[:, 1].astype(np.int64); ea = events[:, 2]
    outside = (ef < start) | (ef >= end)
    unmapped = ~np.isin(ec, channels)
    duplicate_count = len(events) - len(np.unique(np.column_stack([ef, ec]), axis=0))
    qc.update(total_events=len(events), mapped_channels=len(channels),
              events_outside_recording=int(outside.sum()), unmapped_events=int(unmapped.sum()),
              duplicate_events=duplicate_count, nonfinite_amplitudes=int((~np.isfinite(ea)).sum()),
              event_frame_order_violations=int(np.sum(np.diff(ef) < 0)))
    for failed, label in [(outside.any(), 'events_outside_recording'), (unmapped.any(), 'unmapped_events'),
                          (duplicate_count > 0, 'duplicate_spike_frame_channel'),
                          (not np.isfinite(ea).all(), 'nonfinite_amplitude'),
                          (np.any(np.diff(ef) < 0), 'event_frame_order_violation')]:
        if failed:
            errors.append(label)
    qc['structural_passed'] = not errors
    if errors:
        return result
    width = int(round(window_seconds * rate)); guard = int(math.ceil(guard_seconds * rate))
    if width < 1 or abs(width / rate - window_seconds) > 1e-9:
        raise ValueError('window must contain integral device frames')
    lo, hi = start + guard, end - guard
    if hi <= lo:
        errors.append('empty_guarded_recording'); qc['structural_passed'] = False; return result
    full_count = (hi - lo) // width; full_end = lo + full_count * width
    qc.update(guard_seconds=guard_seconds, guarded_interval=[lo, hi], full_window_interval=[lo, full_end],
              full_window_seconds=full_count*window_seconds, unused_tail_seconds=(hi-full_end)/rate,
              excluded_guard_events=int(np.sum((ef < lo) | (ef >= hi))),
              excluded_incomplete_tail_events=int(np.sum((ef >= full_end) & (ef < hi))),
              observed_active_channels=int(len(np.unique(ec))),
              event_rate_hz=len(events)/duration,
              mean_abs_event_amplitude_uv=float(np.mean(np.abs(ea))*lsb*1e6) if len(ea) else None)
    windows = []
    for i in range(full_count):
        left, right = lo + i*width, lo + (i+1)*width
        a, b = np.searchsorted(ef, [left, right], side='left')
        f, c, amp = ef[a:b], ec[a:b], ea[a:b]
        # Fixed device-frame bins; distinct channels, not repeated same-channel spikes.
        # Exact rational device-frame arithmetic also works for low-rate fixtures.
        # For 20 kHz this is equivalent to integer division by 200 frames.
        bins = (f-left) * 100 // int(rate)
        pair = np.unique(np.column_stack([bins, c]), axis=0)
        distinct = np.bincount(pair[:, 0], minlength=math.ceil(window_seconds/.01)) if len(pair) else np.array([])
        coincidence = int(np.sum(distinct[bins] >= 2)) if len(bins) else 0
        windows.append(dict(window_index=i, start_frame=left, end_frame=right,
                            time_seconds=(left-start)/rate, duration_seconds=window_seconds,
                            spike_count=len(f), rate_hz_per_channel=len(f)/(len(channels)*window_seconds),
                            active_channel_fraction=len(np.unique(c))/len(channels),
                            coincident_event_fraction=coincidence/len(f) if len(f) else 0.,
                            mean_abs_amplitude_uv=float(np.abs(amp).mean()*lsb*1e6) if len(amp) else None))
    result['windows'] = windows
    result['candidates'] = calibrate_candidates(windows, window_seconds)
    return result
