"""Choose three stimulation sites inside active tissue and build their route in software.

Runs on the acquisition host after quiet_baseline.py. Nothing is downloaded to
the chip and nothing is stimulated: the route is planned on a server-side array
object and saved to STAGE as sample-route.cfg and session-route.json.

    python route_sites.py STAGE --electrodes-per-site 5

A site is its centre electrode plus the nearest electrodes within 40 um that can
each get a stimulation unit of their own. Electrodes that cannot are replaced by
the next nearest, and a site falls back to fewer electrodes rather than fail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

from oddball2 import design, sites as geometry

MAX_ATTEMPTS = 8           # routings per pool
REPLACING_ATTEMPTS = 3     # after these, a failed electrode is dropped without a substitute
MIN_ROUTED = 51


def survey(stage):
    """Everything decided from the closed quiet baseline; contacts nothing."""
    from mea_stim_v2.recording import H5Recording
    from select_sites import candidates
    b = json.loads((stage / 'baseline-binding.json').read_text(encoding='utf-8'))
    source = Path(b['record_root']) / (b['session_id'] + '-site-selection')
    original = json.loads((source / 'site-candidates.json').read_text(encoding='utf-8'))
    with H5Recording(source / 'quiet.raw.h5', SimpleNamespace(threshold_h5_dataset=b['threshold_h5_dataset'])) as rec:
        measured = candidates(rec, b['chip_id'], b['bad_channels'])
        if original != measured or rec.events or rec.metadata['frame_gaps']:
            raise ValueError('Original site selection did not reproduce on this chip')
        active = geometry.active_electrodes(rec, b['bad_channels'])
        recorded = sorted(e for _, e in rec.mapping)
    chosen = geometry.choose(measured['candidates'], list(active.values()))
    return b, measured, chosen, active, recorded


def wanted_electrodes(centres, per_site, blocked, dropped):
    """Centre first, then the nearest neighbours not yet found unroutable."""
    out = {}
    for site in design.SITES:
        spare = [e for e in geometry.neighbours(centres[site]) if e not in blocked]
        out[site] = [centres[site]] + spare[:max(0, per_site - 1 - dropped[site])]
    return out


def attempt(array, mx, pool, wanted):
    """Route once and connect the wanted electrodes in session order.

    Returns the electrode -> channel mapping, electrode -> unit for the
    electrodes that got a unit of their own, and the electrodes that did not.
    """
    flat = [e for s in design.SITES for e in wanted[s]]
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(sorted(set(pool) | set(flat)))
    array.select_stimulation_electrodes(flat)
    array.route()
    for unit in range(32):
        channel = array.query_amplifier_at_stimulation(unit).strip()
        if channel not in ('', '-1'):
            array.disconnect_amplifier_from_stimulation(int(channel))
    units, failed = {}, []
    for e in flat:
        try:
            array.connect_electrode_to_stimulation(e)
            value = array.query_stimulation_at_electrode(e).strip()
        except RuntimeError:
            value = ''
        if value.isdigit() and 0 <= int(value) < 32 and int(value) not in units.values():
            units[e] = int(value)
        else:
            failed.append(e)
    config = mx.send_raw('mea_array_command ' + array.token + '\nmea_array_get_config').strip()
    mapping = {int(e): int(c) for c, e in re.findall(r'(\d+)\((\d+)\)', config.split()[0] if config else '')}
    for e, unit in list(units.items()):
        if e not in mapping or array.query_amplifier_at_stimulation(unit).strip() != str(mapping[e]):
            failed.append(e)
            del units[e]
    return mapping, units, failed


def plan(stage, per_site):
    import fcntl
    from livepair.device import write_once
    from mea_stim_v2.live import mx, _server_snapshot
    from vendor.brain_stimulation import transport
    if type(per_site) is not int or not 1 <= per_site <= design.MAX_SITE_ELECTRODES:
        raise ValueError('Electrodes per site must be 1 to %d' % design.MAX_SITE_ELECTRODES)
    contract, cfg = stage / 'session-route.json', stage / 'sample-route.cfg'
    if contract.exists() or cfg.exists():
        raise ValueError('This stage directory already holds a route')
    b, measured, chosen, active, recorded = survey(stage)
    centres = {s: chosen['sites'][s]['electrode'] for s in design.SITES}
    centre_xy = {s: geometry.electrode_xy(e) for s, e in centres.items()}
    pools = [set(active) | set(r['electrode'] for r in measured['candidates']) | set(recorded[::5]), set(recorded)]
    lease = Path.home() / '.local/state/mea_photo_full/device.lock'
    with lease.open('a') as guard, transport.bind_sdk(mx, transport.Endpoint(timeout=90)):
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _server_snapshot()[1] or mx.send_raw('wellplate_query_version').strip() != '0':
            raise ValueError('Instrument busy or not MaxOne')
        for strategy, pool in enumerate(pools):
            array = mx.Array('design_' + uuid.uuid4().hex)
            try:
                blocked, dropped = set(), {s: 0 for s in design.SITES}
                for number in range(MAX_ATTEMPTS):
                    wanted = wanted_electrodes(centres, per_site, blocked, dropped)
                    mapping, units, failed = attempt(array, mx, pool, wanted)
                    near = geometry.surroundings(centre_xy, {e: xy for e, xy in active.items() if e in mapping})
                    if (len(mapping) < MIN_ROUTED or any(centres[s] in failed for s in design.SITES)
                            or any(len(near[s]) < geometry.MIN_NEAR for s in design.SITES)):
                        break
                    if failed:
                        blocked |= set(failed)
                        if number + 1 >= REPLACING_ATTEMPTS:
                            for s in design.SITES:
                                dropped[s] += sum(e in failed for e in wanted[s])
                        continue
                    for e in units:
                        array.disconnect_amplifier_from_stimulation(mapping[e])
                    with cfg.open('x') as handle:
                        handle.write(mx.send_raw('mea_array_command ' + array.token + '\nmea_array_get_config').strip())
                    described = {s: dict(centre=centres[s], electrodes=wanted[s],
                                         coordinates=[list(geometry.electrode_xy(e)) for e in wanted[s]],
                                         near_active=chosen['sites'][s]['near_active'], routed_near=len(near[s]))
                                 for s in design.SITES}
                    write_once(contract, dict(
                        chip_id=b['chip_id'], organoid_id=b['organoid_id'], session_id=b['session_id'],
                        baseline=measured['source'], sites=described, electrodes=wanted,
                        software_units={s: [units[e] for e in wanted[s]] for s in design.SITES},
                        route_file=str(cfg), route_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
                        channels=len(mapping), strategy=strategy, attempts=number + 1,
                        electrodes_per_site=per_site, unroutable=sorted(blocked),
                        selection={k: chosen[k] for k in ('distances_um', 'qualified_centres', 'considered_centres', 'rule')},
                        hardware_downloaded=False))
                    print(contract.read_text(encoding='utf-8'), flush=True)
                    for s in design.SITES:
                        print('site %s: centre %d at (%.1f, %.1f) um, %d electrodes, %d active channels within %d-%d um'
                              % (s, centres[s], *centre_xy[s], len(wanted[s]), len(near[s]), *geometry.NEAR_UM), flush=True)
                    return
            finally:
                array.close()
    raise ValueError('The three chosen sites could not be routed with enough active channels around them')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', type=Path)
    p.add_argument('--electrodes-per-site', type=int, required=True, choices=range(1, design.MAX_SITE_ELECTRODES + 1),
                   help='electrodes driven together at each site; agreed with the person responsible for the sample')
    args = p.parse_args()
    plan(args.stage, args.electrodes_per_site)
