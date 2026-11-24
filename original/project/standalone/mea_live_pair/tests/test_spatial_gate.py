import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from mea_state2_r2.features import digest,fit
from livepair.protocol import window


class SpatialTests(unittest.TestCase):
    def test_best_spread_in_three_regions_precedes_rate(self):
        from select_sites import spatial_choice
        pts=[(0,0),(1,0),(2,0),(1000,0),(0,1000),(1000,1000)]
        rows=[dict(channel=i,electrode=100+i,x_um=x,y_um=y,min_spikes_per_20s=100-i)
              for i,(x,y) in enumerate(pts)]
        choice=spatial_choice(rows,(500,500))
        self.assertEqual(choice['region_count'],3)
        self.assertEqual(choice['minimum_distance_um'],1000)
        self.assertEqual(len(choice['distances_um']),3)
        self.assertEqual([choice['suggested'][k]['channel'] for k in 'HWF'],[0,3,4])
        close=spatial_choice(rows[:3],(500,500))
        self.assertEqual((close['region_count'],close['minimum_distance_um']),(1,1))

    def test_missing_or_ambiguous_coordinates(self):
        from select_sites import coordinates
        for xy in (None,{0:(1.,1.),1:(1.,1.)},{0:(1.,0.),1:(float('nan'),1.)}):
            with self.subTest(xy=xy),self.assertRaises(ValueError):
                coordinates(SimpleNamespace(channel_xy_um=xy,channels=np.array([0,1])))

    def test_decoding_and_pre_window_control(self):
        from livepair.qualification import classify_blocks, assess, order_for
        labels=order_for('d'*64)
        self.assertEqual([sorted(x) for x in labels],[sorted('HWF')]*3)
        signals=[[[float(k==s) for k in 'HWF']+[0.] for s in b] for b in labels]
        flat=np.zeros((3,3,4)).tolist()
        ok=assess(labels,signals,flat)
        self.assertTrue(ok['passed'])
        self.assertEqual(ok['post']['correct'],9)
        self.assertAlmostEqual(ok['post']['p_exact'],1/36)
        self.assertFalse(assess(labels,signals,signals)['passed'])
        self.assertFalse(assess(labels,flat,flat)['passed'])
        self.assertEqual(classify_blocks(labels,flat)['confusion'][0], [0,0,0,3])
        with self.assertRaises(ValueError):assess(labels[:2],signals[:2],flat[:2])

    def test_stale_device_and_failed_screen_never_open_gpu(self):
        from control import run
        from livepair.qualification import VERSION
        b=dict(chip_id='c',organoid_id='o',session_id='s',device_hostname='h',replicate=0,
               device_model='MaxOne',acquisition_host='192.0.2.21',device_launch='launch',
               started_at='2026-11-18T12:00:00+08:00')
        for result in ({'sites':list('HWF')},{'passed':False,'binding':digest(b)}):
            peer=Mock()
            peer.side_effect=[dict(qualification=VERSION),dict(binding=digest(b)),result]
            with TemporaryDirectory() as folder,patch('control.Peer',return_value=peer) as connect:
                with self.assertRaises((ValueError,RuntimeError)):
                    run(b,{},Path(folder)/'trial')
                self.assertEqual(connect.call_count,1)
                self.assertEqual(connect.call_args.args[3],'device')
                self.assertTrue((Path(folder)/'trial'/'qualification.json').is_file())
                peer.close.assert_called_once_with(device=True)
        peer=Mock(return_value={})
        with TemporaryDirectory() as folder,patch('control.Peer',return_value=peer):
            with self.assertRaises(ValueError):run(b,{},Path(folder)/'trial')
            peer.assert_called_once_with('protocol')

    def test_nine_closed_h5_trials_through_live_screen(self):
        import h5py
        from test_live_pair import synthetic_h5
        from livepair.device_rpc import Session
        from livepair.qualification import require_pass
        from livepair.readout import admit
        from mea_stim_v2.recording import H5Recording
        with TemporaryDirectory() as folder:
            root=Path(folder)
            ctx=Session()
            d=SimpleNamespace(bid='b'*64,run_dir=root,attempts=0,response_passed=False,
                              norm=fit([[0.,0.,0.,0.]]),cleanup=Mock(return_value=[]))
            ctx.dev=d
            used=[]
            def slot(q,calibration):
                self.assertTrue(calibration)
                used.append(q['site'])
                p=root/('trial%d.raw.h5'%len(used))
                synthetic_h5(p,q)
                offset=len(used)*600000
                with h5py.File(p,'r+') as f:
                    g=f['data_store/data0000']
                    frames=g['groups/routed/frame_nos']; frames[:]=frames[:]+offset
                    ev=g['events'][:];ev['frameno']+=offset;g['events'][:]=ev
                    if q['site'] is not None:
                        d.attempts+=1
                        target=50+'HWF'.index(q['site'])
                        if target!=50:
                            for i in range(3):
                                k=100000+200*i
                                g['groups/routed/raw'][target,k:k+3]=800
                                g['groups/routed/raw'][50,k:k+3]=512
                    # Distinct post responses from independent channels. All pre-windows are silent.
                    spikes={'H':[(340500,2)],'W':[(340500,2+i) for i in range(20)],
                            'F':[(340100+1800*i,2) for i in range(10)],None:[]}[q['site']]
                    dt=g['spikes'].dtype;del g['spikes']
                    g['spikes']=np.asarray([(k+offset,c) for k,c in spikes],dtype=dt)
                with H5Recording(p,SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')) as rec:
                    return admit(rec,q,20,list(range(50)),dict(H=50,W=51,F=52))
            d.slot=slot
            report=ctx.calibrate()
            require_pass(report,d.bid)
            self.assertTrue(d.response_passed)
            self.assertEqual(used.count(None),1)
            self.assertEqual([used.count(s) for s in 'HWF'],[3]*3)
            self.assertEqual(report['analysis']['post']['correct'],9)
            with self.assertRaises(ValueError):ctx.calibrate()

    def test_incomplete_calibration_never_releases_training(self):
        from livepair.device_rpc import Session
        with TemporaryDirectory() as folder:
            ctx=Session()
            ctx.dev=SimpleNamespace(bid='1'*64,run_dir=Path(folder),attempts=0,response_passed=False,
                    slot=Mock(side_effect=ValueError('indeterminate output')),cleanup=Mock(return_value=[]))
            report=ctx.calibrate()
            self.assertFalse(report['passed'])
            ctx.dev.cleanup.assert_called_once()
            with self.assertRaises(ValueError):ctx.prime('prime_train')
