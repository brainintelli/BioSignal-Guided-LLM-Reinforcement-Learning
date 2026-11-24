import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

from livepair import predictability as feedback
from livepair.protocol import GAMMA, reward
from livepair.readout import admit
from mea_stim_v2.recording import H5Recording
from test_live_pair import synthetic_h5


def transition(low=85, gain=1):
    before=dict(tick=0, visited_count=6, health=90, hunger=90, thirst=90, dead=False)
    t1=dict(before,tick=1,thirst=low)
    after=dict(before,tick=2,visited_count=6+gain)
    return dict(before=before, after=after, trace=[t1,after], elapsed_ticks=2, new_discovered=gain)


def req(t, i=2):
    return feedback.request('a'*64, 'ordered', 'train', 0, i, 'b'*64, t)


class PredictabilityTests(unittest.TestCase):
    def test_worker_action_to_reward(self):
        from livepair.worker import Worker
        from livepair.protocol import Buffer, RATE, window
        from mea_state2_r2.features import digest
        # A scripted policy and game exercise the worker's real transport path.
        with tempfile.TemporaryDirectory() as d:
            w=Worker()
            w.config=dict(feedback_encoding=feedback.MODE, arm='ordered', seed=55, binding_sha256='b')
            w.rows=[];w.episode=0;w.out=Path(d);w.phase='train'
            w.buffer=Buffer('ordered',55,[window('b',i*RATE,[0]*4,'prime') for i in range(4)],'b')
            w.rt=Mock();w.prepared=Mock(return_value={})
            w.rt.act.return_value=dict(action='move')
            for i in range(6):
                w.rt.step.return_value=dict(reward=.25, duration=2, transition=transition(gain=int(i<3)),
                                            terminal=False, observation={'turn':i+1})
                q=w.act()['request'];a=(i+1)*400000
                events=[[site,a+off] for site,off in q['events']]
                receipt=dict(request=q, status='accepted', verified_site_frames=events,
                             frames=[f for _,f in events], observed=q['pattern'], anchor=a,
                             window=window('b',a+240000,[0]*4,q['id']))
                w.accept(dict(receipt,id=digest(receipt)))
            self.assertEqual([r['score']['r2'] for r in w.rows],[0,0,1,0,0,-1])

    def test_game_label_checks_whole_transition(self):
        self.assertTrue(feedback.outcome(transition())['correct'])
        self.assertFalse(feedback.outcome(transition(low=84))['correct'])
        self.assertFalse(feedback.outcome(transition(gain=0))['correct'])
        with self.assertRaises(ValueError):
            feedback.outcome(dict(transition(),trace=[]))

    def test_waveform_and_randomization(self):
        for correct in (True, False):
            q=req(transition(gain=int(correct)))
            events=feedback.verify_request(q)
            self.assertEqual(events[-1][1],8000)
            self.assertEqual(len(set(x[0] for x in events)),1 if correct else 3)
            self.assertEqual(events[1][1]==4000,correct)
            plan=feedback.sequence(q,2.9)
            frame=0
            starts=[]
            for k,v in plan:
                if k=='delay': frame+=v
                if k=='site_dac' and v[1]<512: starts.append([feedback.SITES[v[0]],frame])
            self.assertEqual(starts,events)
            self.assertEqual(frame,8008)
        self.assertEqual(feedback.sequence(req(transition(),0),2.9),[])
        q=req(transition())
        with self.assertRaises(ValueError):feedback.verify_request(dict(q,events=[]))

    def test_closed_h5_receipts(self):
        sites=dict(H=50,W=51,F=52)
        for kind,fail in [('regular',None),('irregular',None),('quiet',None),
                          ('irregular','wrong_site'),('regular','extra'),('quiet','extra'),
                          ('regular','wrong_frame'),('irregular','missing')]:
            q=req(transition(gain=int(kind=='regular')),0 if kind=='quiet' else 2)
            with self.subTest(kind=kind,fail=fail),tempfile.TemporaryDirectory() as d:
                path=Path(d)/'wf.h5'
                synthetic_h5(path,q)
                with h5py.File(path,'r+') as f:
                    store=f['data_store/data0000']
                    raw=store['groups/routed/raw']
                    raw[50,100000:100405]=512
                    anchor=100000
                    last=q['events'][-1][1]+8 if q['pulses'] else 0
                    noise=np.tile(np.array([510,514],dtype='u2'),10000)
                    for off in (40000,200000):
                        raw[:50,anchor+last+off:anchor+last+off+20000]=np.tile(noise,(50,1))
                    ev=store['events']
                    for j,(s,off) in enumerate(q['events'] or [('quiet',0)]):
                        t=anchor+off
                        row=(t+int(fail=='wrong_frame' and j==1),1,20+j,
                             json.dumps(dict(tx=q['id'],idx=j,site=s)).encode())
                        ev[j]=row
                        if s!='quiet' and not (fail=='missing' and j==1):
                            chan=sites[s]
                            if fail=='wrong_site' and j==1:chan=50+(chan-50+1)%3
                            raw[chan,t:t+4]=800
                    if fail=='extra':raw[51,anchor+9000:anchor+9003]=800
                cfg=SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
                with H5Recording(path,cfg) as rec:
                    if fail:
                        with self.assertRaises(ValueError):admit(rec,q,20,list(range(50)),sites)
                    else:
                        receipt=admit(rec,q,20,list(range(50)),sites)
                        score=reward(1.,2,False,q,receipt)
                        sign=dict(regular=1,irregular=-1,quiet=0)[kind]
                        self.assertEqual(score['r2'],sign)
                        self.assertAlmostEqual(score['total'],1+.001*sign*(1+GAMMA))
