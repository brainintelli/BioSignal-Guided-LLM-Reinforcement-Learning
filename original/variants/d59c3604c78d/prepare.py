"""Offline preparation only. No SDK, network, device or training entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

MAX_SECONDS = 7200
RECOVERY_SECONDS = 600
CLOSEOUT_SECONDS = 120


def number(value, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError('Expected a finite numeric value, not bool/null')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('Invalid numeric input') from exc
    if (not result.is_finite() or result < 0 or result > Decimal('1e12')
            or (positive and result == 0)):
        raise ValueError('Numeric input outside supported range')
    return result


def quantize(p90_uv, lsb_mv):
    """Mathematical nominal encodability; never an output authorization."""
    p90, lsb = number(p90_uv, positive=True), number(lsb_mv, positive=True)
    with localcontext() as ctx:
        ctx.prec = max(60, len(p90.as_tuple().digits) + 20, len(lsb.as_tuple().digits) + 20)
        cap_uv = p90 * Decimal('1.2')
        cap_mv = cap_uv / Decimal(1000)
        ratio = Fraction(cap_mv) / Fraction(lsb)
        offset = min(511, ratio.numerator // ratio.denominator)
        candidate = lsb * offset if offset else None
    return {'p90_uv': str(p90), 'cap_uv': str(cap_uv), 'cap_mv': str(cap_mv),
            'dac_lsb_mv': str(lsb), 'offset_bits': offset,
            'candidate_phase_mv': str(candidate) if candidate is not None else None,
            'nonzero_encodable': offset > 0,
            'physical_output_verified': False,
            'reason': 'nominal_encoding_only' if offset else 'cap_below_one_dac_step'}


def budget(elapsed_seconds, work_seconds=0):
    """Pure planning calculation, NOT a running timer or a hardware stop."""
    elapsed, work = number(elapsed_seconds), number(work_seconds)
    remaining = max(Decimal(0), MAX_SECONDS - elapsed)
    return {'elapsed_seconds': float(elapsed), 'remaining_seconds': float(remaining),
            'latest_work_end_seconds': MAX_SECONDS - RECOVERY_SECONDS - CLOSEOUT_SECONDS,
            'fits_before_recovery': elapsed + work < MAX_SECONDS - RECOVERY_SECONDS - CLOSEOUT_SECONDS,
            'hard_limit_reached': elapsed >= MAX_SECONDS,
            'runtime_enforcement': False}


def prepare(session, reference, source_sha256, *, now=None):
    if (session.get('mode') != 'preparation_only'
            or session.get('execution_enabled') is not False
            or session.get('r2_enabled') is not False):
        raise ValueError('Only preparation_only with execution/R2 disabled is supported')
    if reference.get('schema') != 'mea.p90_peak_cap.feasibility.v1':
        raise ValueError('Unsupported historical reference schema')
    identity = session.get('session')
    if not isinstance(identity, dict):
        raise ValueError('Missing session object')
    blockers = ['historical_reference_not_current_calibration', 'physical_output_unverified',
                'current_electrode_material_mapping_and_route_unverified',
                'runtime_deadline_and_stop_not_implemented',
                'live_state_actor_interface_not_implemented', 'preparation_only_no_executor']
    for key in ('device_host', 'chip_id', 'organoid_id'):
        value = identity.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'{key} must be a nonempty string or null')
        if value is None:
            blockers.append('missing_' + key)
    started = identity.get('started_at')
    timing = {'status': 'not_started', 'clock_started': False, 'runtime_enforcement': False}
    if started is None:
        blockers.append('session_not_started')
    else:
        try:
            start = datetime.fromisoformat(started)
        except (TypeError, ValueError) as exc:
            raise ValueError('started_at must be an ISO timestamp with timezone or null') from exc
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError('started_at requires a timezone')
        observed = now or datetime.now(timezone.utc)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError('observation time requires a timezone')
        elapsed = (observed - start).total_seconds()
        timing = {'status': 'snapshot_only', 'clock_started': False,
                  'observed_at': observed.isoformat(), **budget(elapsed)}
        if not timing['fits_before_recovery']:
            blockers.append('time_budget_does_not_allow_new_output')

    chip = session.get('reference_chip_id')
    rows, seen = [], set()
    for row in reference.get('limits', []):
        if row.get('chip_id') != chip or row.get('scope') != 'per_electrode':
            continue
        electrode = row.get('electrode_id')
        if type(electrode) is not int or not 0 <= electrode <= 26399 or electrode in seen:
            raise ValueError('Invalid or duplicate physical electrode ID')
        seen.add(electrode)
        rows.append({'reference_chip_id': chip, 'electrode_id': electrode,
                     **quantize(row['p90_uv'], row['device_reported_lsb_mv'])})
    if not rows:
        raise ValueError('No per-electrode references for requested historical chip')
    if not any(r['nonzero_encodable'] for r in rows):
        blockers.insert(0, 'no_nonzero_voltage_under_1p2_p90_cap')
    return {'schema': 'mea.biocompute.preparation.v1', 'status': 'blocked_for_execution',
            'execution_enabled': False, 'stimulation_sent': False, 'hardware_connected': False,
            'r2_enabled': False, 'vendor_configuration': None,
            'current_session': dict(identity), 'reference_chip_id': chip,
            'reference_role': 'historical_design_only', 'reference_sha256': source_sha256,
            'peak_multiplier': '1.2', 'electrodes': rows, 'blockers': blockers,
            'timing': timing,
            'limits': {'max_duration_seconds': MAX_SECONDS, 'final_recovery_seconds': RECOVERY_SECONDS,
                       'closeout_seconds': CLOSEOUT_SECONDS, 'planned_seconds': 5520,
                       'reserve_seconds': 1680, 'latest_work_end_seconds': 6480,
                       'future_true_pulse_budget': 25, 'future_sham_budget': 12,
                       'current_authorized_nonzero_pulses': 0},
            'new_organoid_requires_new_baseline': True,
            'hardware_deadline_enforced': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        source_bytes = args.reference.read_bytes()
        session_bytes = args.session.read_bytes()
        report = prepare(json.loads(session_bytes.decode('utf-8')),
                         json.loads(source_bytes.decode('utf-8'), parse_float=Decimal),
                         hashlib.sha256(source_bytes).hexdigest())
        report['session_template_sha256'] = hashlib.sha256(session_bytes).hexdigest()
        report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['source_path'] = str(args.reference.resolve())
        report['session_template_path'] = str(args.session.resolve())
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write('\n')
        print('Preparation report written; hardware execution remains blocked.')
        return 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f'Preparation failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
