import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from mea_binary_r2.encoding import Encoder, Intent
from mea_online.m9_runtime import project_observation,validate_runtime_config
from mea_state2_r2.features import fit,digest
from mea_stim_v2.recording import H5Recording
from livepair.protocol import *
from livepair.readout import admit
from livepair.study import run_pair

BID='a'*64


def req(i=2,site='H',arm='ordered',phase='train',episode=0):
    it=Intent(site,3 if site else 0,True,(site,) if site else ())
    return transaction(BID,arm,phase,episode,i,'b'*64,it)


def synthetic_h5(path,q,failure=None):
    anchor=100000
    n=q['pulses']
    with h5py.File(path,'w') as f:
        f['wellplate/id']=[b'fixture']
        d=f.create_group('data_store/data0000')
        for k,v in dict(sampling=20000,gain=512,hpf=300,lsb=6e-6,spike_threshold=5).items():
            d['settings/'+k]=[v]
        d['settings/mapping']=np.array([(c,c+100) for c in range(53)],dtype=[('channel','i4'),('electrode','i4')])
        d['groups/routed/channels']=np.arange(1024)
        frames=np.arange(370000,dtype='u8')
        if failure=='gap':frames[120000]+=1
        d['groups/routed/frame_nos']=frames
        raw=d.create_dataset('groups/routed/raw',(1024,370000),dtype='u2',chunks=(1,4000),
                             fillvalue=512,compression='gzip')
        for t in (anchor-60000,anchor+(408 if n else 0)+40000,anchor+(408 if n else 0)+200000,anchor+240000):
            noise=np.tile(np.array([510,514],dtype='u2'),10000)
            if failure=='recovery' and t==anchor+240000:noise[:]=1023
            raw[0:50,t:t+RATE]=np.tile(noise,(50,1))
        ed=np.dtype([('frameno','u8'),('eventtype','i4'),('eventid','i4'),('eventmessage','S200')])
        d['events']=np.array([(anchor+200*i,1,20+i,json.dumps(dict(tx=q['id'],idx=i)).encode())
                              for i in range(max(1,n))],dtype=ed)
        for i in range(3 if n or failure=='unexpected' else 0):
            if failure!='missing' or i!=1:raw[50,anchor+200*i:anchor+200*i+3]=800
        d['spikes']=np.array([(anchor+240100,1),(anchor+245000,2),(anchor+240015,50)],
                             dtype=[('frameno','u8'),('channel','i4')])


class CoreTests(unittest.TestCase):
    def test_worker_lifecycle_with_recorded_observation(self):
        from livepair.worker import Worker
        from compare import lower95
        from tests.test_mea_p0 import observation
        obs=observation()
        norm=fit([[1.,0.,0.,0.]])
        cfg=dict(arm='ordered',seed=57,devices=list(range(8)),binding_sha256=BID,normalization=norm,
                 normalization_id=digest(norm),experiment=PROTOCOL)
        class Runtime:
            def __init__(self):self.updates=0
            def reset(self,seed):return obs
            def prepare(self,o,s):return project_observation(o,s,cfg)
            def act(self,row):
                return dict(action='case',value=0.,verb=0,target=0,logp=0.,verb_logp=0.,
                            reference_verb=[1.],reference_target=[1.])
            def step(self,a):
                state=dict(health=80,hunger=80,thirst=80,visited_count=3)
                return dict(observation=obs,transition=dict(before=state,after=state),terminal=False,duration=2,reward=.5)
            def update(self,rows,boot,n):
                assert all(r['reward']==r['score']['total'] for r in rows)
                self.updates+=1
                return dict(actor_changed=True,critic_changed=True)
            def checkpoint(self,*args):return dict(manifest_sha256='technical')
            def state(self):return self.updates
            def value(self,r):return 0.
        with tempfile.TemporaryDirectory() as folder:
            w=Worker();w.config=cfg;w.out=Path(folder);w.rt=Runtime()
            for p,phase in enumerate(('train','eval')):
                base=(p+1)*100000000
                init=[window(BID,base+RATE*i,[0]*4,'prime') for i in range(4)]
                w.start(phase,init)
                for i in range(STEPS):
                    q=w.act()['request'];anchor=base+(i+1)*400000
                    ww=window(BID,anchor+240000,[.1,.2,.3,.4],q['id'])
                    r=dict(request=q,anchor=anchor,frames=[anchor+200*j for j in range(q['pulses'])],
                           window=ww,observed='output' if q['pulses'] else 'quiet',status='accepted')
                    w.accept(dict(r,id=digest(r)))
                w.finish()
            self.assertEqual(w.rt.updates,1)
            self.assertEqual(lower95([0,1,2,3,4,5,6,7]),1)
            self.assertIsNone(lower95([1]))

    def test_causal_permutation(self):
        initial=[window(BID,i*RATE,[i,0,0,0],'initial') for i in range(4)]
        c,d=[Buffer(a,715,initial,BID) for a in ARMS]
        for block in range(2):
            seen=[[],[]]
            for i in range(4):
                for j,b in enumerate((c,d)):
                    seen[j].append(b.take()['start'])
                    b.push(window(BID,(4+block*4+i)*RATE,[0]*4,'later'))
            self.assertEqual(sorted(seen[1]),seen[0])
            self.assertTrue(all(a!=b for a,b in zip(*seen)))
        with self.assertRaises(ValueError):c.push(initial[0])

    def test_maximum_and_terminal(self):
        e=Encoder(3)
        intents=[scheduled_intent(e,dict(H=10,W=5,F=9),False,i) for i in range(16)]
        self.assertEqual(sum(i.pulses>0 for i in intents),5)
        self.assertEqual(4*sum(i.pulses for i in intents)+9*3,87)
        self.assertEqual(reward(-3.,3,True,None,None)['total'],-3.)
        with self.assertRaises(ValueError):reward(-3.,3,True,req(),None)

    def test_real_reader_reward_and_rejections(self):
        cfg=SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
        for site,fail in ((None,None),('H',None),('H','missing'),(None,'unexpected'),('H','gap'),('H','recovery')):
            with self.subTest(site=site,fail=fail),tempfile.TemporaryDirectory() as folder:
                q=req(site=site)
                p=Path(folder)/'test.raw.h5'
                synthetic_h5(p,q,fail)
                def run():
                    with H5Recording(p,cfg) as rec:
                        return admit(rec,q,20,list(range(50)),dict(H=50,W=51,F=52))
                if fail:
                    with self.assertRaises(ValueError):run()
                else:
                    rec=run()
                    score=reward(.1,3,False,q,rec)
                    self.assertEqual(score['r2'],-1 if site else 1)
                    self.assertAlmostEqual(score['total'],.1+score['r2']*.001*(1+GAMMA+GAMMA**2))
                    self.assertAlmostEqual(rec['window']['features'][1],2/50)
                    with self.assertRaises(ValueError):reward(.1,3,False,req(5,site),rec)

    def test_pair_flow_and_natural_death(self):
        phases={};counts={};episodes={};audit=[];slots=[]
        def dev(op,**kw):
            if op=='prime':return []
            q=kw['req'];slots.append(q);return q
        def peer(a):
            def call(op,**kw):
                if op=='start':
                    phases[a]=kw['phase'];episodes[a]=kw.get('episode',0)
                    if not episodes[a]:counts[a]=0
                if op=='act':
                    i=counts[a];p=phases[a]
                    dead=(a=='ordered' and i==(6 if p=='train' else 9))
                    return dict(terminal=dead,request=None if dead else req(i,'H' if i%3==2 else None,a,p,episodes[a]))
                if op=='accept':counts[a]+=1
                if op=='finish':return dict(decisions=counts[a],explored=counts[a],death=int(counts[a]<16))
            return call
        val=run_pair(dev,{a:peer(a) for a in ARMS},0,audit.append)
        self.assertEqual(val['results']['train']['ordered']['decisions'],16)
        self.assertEqual(val['results']['eval']['ordered']['decisions'],10)
        self.assertEqual(val['results']['eval']['shuffled']['decisions'],16)
        self.assertEqual([r['arm'] for r in slots[:4]],['ordered','shuffled','shuffled','ordered'])
        self.assertEqual(len(slots),len(audit)-2)

    def test_metrics_and_incomplete_horizon(self):
        before=dict(visited_count=9,health=90,hunger=80,thirst=70)
        after=dict(visited_count=16,health=0,hunger=85,thirst=65)
        row=dict(terminal=True,duration=8,transition=dict(before=before,after=after))
        m=metrics([row])
        self.assertEqual((m['explored'],m['health'],m['hunger_min'],m['death']),(7,0,80,1))
        with self.assertRaises(ValueError):metrics([dict(row,terminal=False)])


if __name__=='__main__':unittest.main()
