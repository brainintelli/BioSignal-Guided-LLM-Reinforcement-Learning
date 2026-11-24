import json
import io
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from select_sites import spatial_choice
from livepair.qualification import VERSION, analysis, order_for
from livepair.protocol import digest


def fixture(correct=True):
    pts=[dict(channel=i,electrode=200+i,x_um=x,y_um=y,min_spikes_per_20s=10-i)
         for i,(x,y) in enumerate([(0.,0.),(600.,0.),(0.,800.)])]
    selection=dict(chip='sample',candidates=pts,**spatial_choice(pts,(300.,400.)))
    b=dict(session_id='offline',chip_id='sample',organoid_id='test',device_hostname='test',
           electrodes={s:r['electrode'] for s,r in selection['suggested'].items()},
           replicate=0,device_model='MaxOne',acquisition_host='192.0.2.21',device_launch='launch',
           started_at='2026-11-18T12:00:00+08:00')
    bid=digest(b)
    order=order_for(bid)
    trials=[]
    for block,sites in enumerate(order):
        for s in sites:
            i=len(trials)
            trials.append(dict(block=block,site=s,index=i,request_id=digest(['q',i]),receipt_id=digest(['r',i]),
                        pre_id=digest(['a',i]),post_id=digest(['b',i]),anchor=i*600000,
                        pre=[0.]*4,post=[float(correct and s==k) for k in 'HWF']+[0.]))
    stats=analysis(bid,trials)
    report=dict(schema=VERSION,binding=bid,order=order,trials=trials,analysis=stats,
                passed=stats['passed'],attempts=9,quiet_receipt='f'*64)
    return b,selection,report


class PlotTests(unittest.TestCase):
    def test_presentation_cue_requires_delivery_and_leaves_failed_runs_stopped(self):
        from control import present_in_chat
        with patch('control.announce') as announce:
            with patch('control.sys.stdin',io.StringIO('')):
                with self.assertRaises(RuntimeError):present_in_chat({'passed':True},time.time())
                present_in_chat({'passed':False},time.time())
            with patch('control.sys.stdin',io.StringIO('shown\n')):
                present_in_chat({'passed':True},time.time())
            self.assertEqual(announce.call_count,3)

    def test_complete_failed_and_interrupted_figures(self):
        from livepair.figures import draw_sites,draw_responses
        from PIL import Image
        with TemporaryDirectory() as folder:
            d=Path(folder)
            b,s,_=fixture()
            with Image.open(draw_sites(d,b,s)) as image:
                self.assertGreater(image.width,800)
            for passed in (True,False):
                b,_,report=fixture(passed)
                with Image.open(draw_responses(d,b,report)) as image:
                    self.assertGreater(image.width,800)
            report.pop('analysis')
            report['trials']=report['trials'][:4]
            report['error']='synthetic interrupted capture'
            with Image.open(draw_responses(d,b,report)) as image:
                self.assertGreater(image.width,800)
            s['distances_um']['H-W']+=5
            with self.assertRaises(ValueError):draw_sites(d,b,s)
            report['binding']='0'*64
            with self.assertRaises(ValueError):draw_responses(d,b,report)

    def test_figures_presented_before_gpu_and_failure_stops(self):
        from control import run
        from contextlib import nullcontext
        for passed in (True,False):
            b,selection,report=fixture(passed)
            order=[]
            device=Mock()
            device.side_effect=[dict(qualification=VERSION),dict(binding=digest(b),out='/data/offline',norm={}),report]
            device.selection.return_value=selection
            gpu=Mock(return_value={'initial':'identical'})
            def connect(stack,host,launch,role,log):
                order.append(role)
                return device if role=='device' else gpu
            def present(figures):
                self.assertEqual(figures['passed'],passed)
                for path in figures['files']:
                    self.assertTrue(Path(path).is_file())
                order.append('figures')
            hosts={h:dict(launch='launch',project='p',godot='g',model_paths={})
                   for h in ('192.0.2.16','192.0.2.23')}
            guard=nullcontext() if passed else self.assertRaises(ValueError)
            with TemporaryDirectory() as folder,patch('control.Peer',side_effect=connect),patch('control.run_pair',return_value={}):
                out=Path(folder)/'run'
                with guard:run(b,hosts,out,present=present)
                self.assertEqual(order,['device','figures']+(['gpu','gpu'] if passed else []))
                self.assertEqual(json.loads((out/'spatial-selection.json').read_text())['chip'],b['chip_id'])
