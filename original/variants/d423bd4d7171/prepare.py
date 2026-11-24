"""Register an earlier, untrained recording with a different route through the same electrodes."""
from pathlib import Path
import csv,hashlib,json,random
import h5py
import numpy as np
from mea_state2_r2.features import digest,transform
import sys
ROOT=Path(__file__).resolve().parent
TRAIN=ROOT.parent
sys.path.insert(0,str(TRAIN))
from build_inputs import window,write,read,FS
SESSION=TRAIN.parent
CROSS=SESSION/'cpp-crossover-v2'

def main():
    source=SESSION/'baseline/quiet.raw.h5'
    warm=CROSS/'warmup/warmup.raw.h5'
    site=read(SESSION/'baseline/site-candidates.json')
    binding=read(SESSION/'baseline-binding.json')
    ctx=read(TRAIN/'pre-stim-baseline/baseline.json')
    spec=read(TRAIN/'design.json')
    assert site['chip']==binding['chip_id']==spec['chip']=='22021' and binding['organoid_id']==ctx['sample']
    hs=read(SESSION/'baseline/verified.json')
    with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==hs['sha256'] and source.stat().st_size==hs['bytes']
    with warm.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==read(CROSS/'raw-hashes.json')['warmup/warmup.raw.h5']['sha256']
    roi=read(CROSS/'design.json')['roi']
    assert len(set(roi))==322 and digest(roi)==ctx['roi_sha']
    with h5py.File(source) as f, h5py.File(warm) as w:
        old=f['data_store/data0000']; route=w['data_store/data0000']
        for key in ('gain','hpf','sampling','lsb','spike_threshold'):
            assert np.array_equal(old['settings/'+key][:],route['settings/'+key][:])
        assert float(old['settings/sampling'][0])==FS and len(old['events'])==0
        oldmap=old['settings/mapping'][:];routemap=route['settings/mapping'][:]
        old_by_el={int(r['electrode']):r for r in oldmap};new_by_ch={int(r['channel']):r for r in routemap}
        assert len(old_by_el)==len(oldmap) and len(new_by_ch)==len(routemap)
        native_to_roi={};electrodes=[]
        for ch in roi:
            target=new_by_ch[ch]; electrode=int(target['electrode']);orig=old_by_el[electrode]
            assert all(target[k]==orig[k] for k in ('x','y'))
            native_to_roi[int(orig['channel'])]=ch;electrodes.append(electrode)
        assert len(native_to_roi)==len(set(electrodes))==322
        native=old['spikes'][:];sp=native[np.isin(native['channel'],list(native_to_roi))].copy()
        sp['channel']=[native_to_roi[int(ch)] for ch in sp['channel']]
        frames=old['groups/routed/frame_nos'][:].astype('int64')
        assert np.all(np.diff(frames)==1) and frames[-1]<int(route['groups/routed/frame_nos'][0])
        first,last=map(int,(frames[0],frames[-1]));wins=[]
        for idx in range(60):
            s=idx+2
            z=window(sp,first,last,first+s*FS,20000+idx,0,s,roi)
            z['normalized']=list(transform(z['features'],ctx['normalization']));wins.append(z)
    prior=set()
    for f in TRAIN.parents[1].rglob('design.json'):
        if ROOT in f.parents:continue
        obj=read(f)
        if isinstance(obj,dict):
            prior.update(i for i in obj.get('holdout_seeds',[]) if isinstance(i,int))
            ep=obj.get('episodes',{})
            if isinstance(ep,dict):prior.update(v for seq in ep.values() if isinstance(seq,list) for v in seq if isinstance(v,int))
    worlds=random.Random(202611182207).sample(range(177000000,178000000),8)
    assert not set(worlds)&(prior|set(spec['training_seeds']))
    order=list(range(60)); shuffle={}
    for world in worlds:
        rand=random.Random(world+22021)
        while True:
            blocks=rand.sample(range(12),12)
            if all(i!=j for i,j in enumerate(blocks)):break
        shuffle[str(world)]=[5*b+j for b in blocks for j in range(5)]
        assert sorted(shuffle[str(world)])==order
    modes=['trained_ordered','trained_shuffled','trained_masked','control_masked','initial_ordered']
    reg=dict(session=binding['session_id'],sample=ctx['sample'],chip=binding['chip_id'],source_sha=hs['sha256'],
        source_file='baseline/quiet.raw.h5',normalization=ctx['normalization'],roi_sha=ctx['roi_sha'],
        remap=[[int(old),int(new)] for old,new in native_to_roi.items()],aligned_electrodes=electrodes,
        source_frames=[first,last],windows=wins,worlds=worlds,shuffle=shuffle,seeds=spec['training_seeds'],
        clock='One archived second per game tick, indexed at action start. No wrap. Terminal state read at tick 60 or earlier death.',
        horizon=60,scenes=spec['starting_reserves'],modes=modes,study_note='Earlier baseline used for electrode selection, not PPO training or normalization. Secondary archived evaluation.',
        primary_comparisons=[['trained_ordered','trained_shuffled'],['trained_ordered','control_masked'],['trained_ordered','initial_ordered']],
        diagnostic_comparison=['trained_ordered','trained_masked'],endpoints=['survived','r1','new_cells'],primary_endpoint='survived',
        bootstrap=dict(seed=207607,draws=30000),parent_sha=digest(spec))
    write(ROOT/'design.json',reg)
    print('same physical electrodes',len(electrodes),'new windows',len(wins),'unique',len(set(tuple(w['normalized']) for w in wins)),'prior games checked',len(prior),'source seconds',(last-first+1)/FS)
if __name__=='__main__':main()
