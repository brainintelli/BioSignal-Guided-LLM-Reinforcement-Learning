"""The routed sites of one MaxOne chip, through the audited MaxLab backend.

Each site is one to five neighbouring electrodes. Every electrode has its own
stimulation unit, and all units of a site take the same DAC channel, so the
electrodes of a site always receive the same pulse at the same instant.
Importing this module contacts nothing.
"""
import hashlib
from pathlib import Path
import re
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

from livepair.protocol import binding
from mea_stim_v2 import live
from mea_stim_v2.config import route_mapping
from mea_stim_v2.recording import H5Recording

from . import design, sites as geometry

mx = live.mx
XY_TOLERANCE_UM = .5


class Route:
    """The bound route file and the electrodes of each site, centre first."""

    def __init__(self, b):
        self.b = b
        self.device_hostname = b['device_hostname']
        self.route_file = b['route_file']
        self.session_id = b['session_id']
        self.threshold_h5_dataset = b['threshold_h5_dataset']
        self.sites = b['electrodes']
        if set(self.sites) != set(design.SITES):
            raise ValueError('Electrodes for exactly the sites H, W and F required')
        flat = [e for s in design.SITES for e in self.sites[s]]
        if (any(not 1 <= len(self.sites[s]) <= design.MAX_SITE_ELECTRODES for s in design.SITES)
                or any(type(e) is not int for e in flat) or len(set(flat)) != len(flat)):
            raise ValueError('One to %d distinct electrodes per site required' % design.MAX_SITE_ELECTRODES)
        self.stim_electrode = self.sites['H'][0]
        self.validate_binding()

    def validate_binding(self):
        if hashlib.sha256(Path(self.route_file).read_bytes()).hexdigest() != self.b['route_sha256']:
            raise ValueError('Frozen route changed')
        pairs = route_mapping(self.route_file)
        routed = set(dict(pairs).values())
        if any(e not in routed for s in design.SITES for e in self.sites[s]):
            raise ValueError('Selected electrode absent from route')
        return pairs

    def validate_live(self):
        binding(self.b)
        if self.b['device_model'] != 'MaxOne' or self.b['material'] != 'Pt-Black':
            raise ValueError('Unsupported chip')


class Device(live.MaxLabBackend):
    def __init__(self, b, run_dir):
        self.b = b
        self.start = binding(b)
        addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
        if b['acquisition_host'] not in addresses or socket.gethostname() != b['device_hostname']:
            raise RuntimeError('Wrong acquisition host')
        super().__init__(Route(b), run_dir, SimpleNamespace(wall=time.time, mono=time.monotonic, sleep=time.sleep))
        self.units = {}     # site -> stimulation units, in the order of the site's electrodes
        self.sites = {}     # site -> amplifier channels, in the same order

    def _pairs(self):
        return [(site, e, unit) for site in design.SITES for e, unit in zip(self.config.sites[site], self.units[site])]

    def _check_mapping(self):
        raw = mx.send_raw('mea_array_command ' + self.array_token + '\nmea_array_get_config')
        first = raw.split()[0] if raw.split() else ''
        pairs = self.config.validate_binding()
        if sorted((int(c), int(e)) for c, e in re.findall(r'(\d+)\((\d+)\)', first)) != pairs:
            raise ValueError('Downloaded route differs')
        channel = {e: c for c, e in pairs}
        links = {}
        for unit in range(32):
            value = self.array.query_amplifier_at_stimulation(unit).strip()
            if value not in ('', '-1'):
                links[unit] = int(value)
        if links != {unit: channel[e] for _, e, unit in self._pairs()}:
            raise ValueError('Unexpected stimulation attachment')
        for _, e, unit in self._pairs():
            if self.array.query_stimulation_at_electrode(e).strip() != str(unit):
                raise ValueError('Electrode/unit association changed')
        return raw

    def prepare_sites(self):
        self._check_server(False)
        with self._sdk():
            self.touched = True
            mx.initialize()
            for unit in range(32):
                mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
            for dac in range(len(design.SITES)):
                mx.send(mx.DAC(dac, 512))
            mx.set_gain(512)
            mx.hpf('300Hz')
            mx.send(mx.Core().enable_stimulation_power(True))
            self.array_token = 'odd2_' + uuid.uuid4().hex
            self.array = mx.Array(self.array_token, persistent=True)
            self.array.load_config(self.config.route_file)
            for unit in range(32):
                value = self.array.query_amplifier_at_stimulation(unit).strip()
                if value not in ('', '-1'):
                    self.array.disconnect_amplifier_from_stimulation(int(value))
            channel = {e: c for c, e in self.config.validate_binding()}
            used = []
            for site in design.SITES:
                self.units[site] = []
                for e in self.config.sites[site]:
                    self.array.connect_electrode_to_stimulation(e)
                    value = self.array.query_stimulation_at_electrode(e).strip()
                    if not value.isdigit() or not 0 <= int(value) < 32 or int(value) in used:
                        raise ValueError('Electrode %d has no stimulation unit of its own' % e)
                    if self.array.query_amplifier_at_stimulation(int(value)).strip() != str(channel[e]):
                        raise ValueError('The unit of electrode %d sits on another amplifier' % e)
                    used.append(int(value))
                    self.units[site].append(int(value))
            mx.activate([0])
            self.array.download([0])
            time.sleep(5)
            mx.offset()
            time.sleep(5)
            raw = self._check_mapping()
        (self.run_dir / 'downloaded.cfg').write_text(raw)
        self.prepared = True
        self.sites = {site: [channel[e] for e in self.config.sites[site]] for site in design.SITES}


class MaxOneRig:
    def __init__(self, b, route, out):
        self.Sequence, self.Recording = live.ManagedSequence, H5Recording
        self.binding, self.route = b, route
        self.dev = Device(b, out)
        self.lsb_mv = None
        self.site_channels = {}

    @property
    def faulted(self):
        return self.dev._faulted

    def prepare(self):
        dev = self.dev
        self.lsb_mv = dev.preflight()['lsb_mv']
        amplitudes = design.ladder(self.binding['ceiling_mv'])
        actual = {str(a): design.pulse_bits(self.lsb_mv, a) * self.lsb_mv for a in amplitudes}
        dev.prepare_sites()
        self.site_channels = {s: list(v) for s, v in dev.sites.items()}
        return dict(units=dev.units, channels=dev.sites, electrodes=dev.config.sites, lsb_mv=self.lsb_mv,
                    actual_mv=actual, software_units=self.route.get('software_units'))

    def open(self, name):
        self.dev.open_record(name)

    def units(self, on):
        with self.dev._sdk():
            for dac in range(len(design.SITES)):
                mx.send(mx.DAC(dac, 512))
            for dac, site in enumerate(design.SITES):
                for unit in self.dev.units.get(site, ()):
                    mx.send(mx.StimulationUnit(unit).power_up(on).connect(on).set_voltage_mode().dac_source(dac))

    def build(self, block, bits):
        """Create every chunk of the block on the server; nothing runs until it is sent."""
        dev = self.dev
        if not dev.file_open or not dev.recording:
            raise RuntimeError('Sequences are built inside an open recording')
        handles = []
        with dev._sdk():
            if float(mx.query_DAC_lsb_mV()) != self.lsb_mv:
                raise RuntimeError('DAC step changed')
            for trials in design.chunks(block):
                token = 'odd2_seq_' + uuid.uuid4().hex
                dev.sequences.append(token)  # owned before the constructor; its ACK may be lost
                sequence = self.Sequence(token=token, initial_delay=100, persistent=True)
                if sequence.token != token:
                    raise RuntimeError('Server returned a different sequence token')
                for kind, value in design.chunk_codes(block, trials, bits):
                    if kind == 'marker':
                        number = design.event_id(block, value)
                        if number in dev._events_attempted:
                            raise ValueError('Event ID already used')
                        dev._events_attempted.add(number)
                        properties = dict(design.event_properties(block, value), session=dev.config.session_id)
                        text = ' '.join('%s %s' % item for item in sorted(properties.items()))
                        sequence.append(mx.Event(0, 1, number, text))
                    elif kind == 'dac':
                        sequence.append(mx.DAC(*value))
                    else:
                        sequence.append(mx.DelaySamples(value))
                handles.append(sequence)
        return handles

    def send(self, handle, event, gate):
        dev = self.dev
        dev.before_sequence_send, dev._pending_event, dev._pending_pulse = gate, dict(event), True
        try:
            with dev._sdk():
                handle.send()
        finally:
            dev.before_sequence_send, dev._pending_event, dev._pending_pulse = None, None, False

    def close(self):
        return self.dev.close_record()['path']

    def read(self, path):
        return self.Recording(path, self.dev.config)

    def verify(self, rec):
        b = self.binding
        expected = dict(chip_id=b['chip_id'], fs=20000., gain=512., hpf=300., frame_gaps=0,
                        threshold_value=b['threshold_value'])
        if any(rec.metadata[k] != v for k, v in expected.items()) or sorted(rec.mapping) != self.dev.config.validate_binding():
            raise ValueError('Chip, recording settings or route differ from the binding')
        for site, channels in self.site_channels.items():
            for e, channel in zip(self.dev.config.sites[site], channels):
                here, wanted = rec.channel_xy_um[channel], geometry.electrode_xy(e)
                if max(abs(here[0] - wanted[0]), abs(here[1] - wanted[1])) > XY_TOLERANCE_UM:
                    raise ValueError('Physical coordinates differ from the selected sites')

    def cleanup(self):
        return self.dev.cleanup()
