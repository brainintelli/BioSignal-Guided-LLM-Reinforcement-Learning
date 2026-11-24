"""Pilot-only empirical location-shift power for joint paired, crossed inference."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PRIOR=HERE/'pilot-paired.csv'
PIN='cc86d8fa2e610c73ad77c771bd88c60282c3fa7ec0f153276bcb25edec88be99'
RNG_SEED=22021119
NS,NW,DRAWS=32,64,40000
ENDPOINTS=[('low_health','cells',1,[3.,5.,8.]),('thirst','survived',1,[.03,.05,.10]),('hunger','survived',1,[.03,.05,.10]),('thirst','drink_wait',-1,[1.,2.,4.]),('hunger','eat_wait',-1,[1.,2.,4.])]


def load_pilot():
    assert sha256(PRIOR.read_bytes()).hexdigest()==PIN
    with PRIOR.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    seeds=sorted({int(r['seed']) for r in rows});maps=sorted({int(r['world']) for r in rows})
    observations={(r['scene'],int(r['seed']),int(r['world'])):r for r in rows}
    assert len(seeds)==8 and len(maps)==64 and len(observations)==len(rows)==1536
    pairs=np.empty((8,64,10),dtype=np.float32)
    for ep,(scene,metric,sgn,_) in enumerate(ENDPOINTS):
        for arm_num,arm in enumerate(['masked','shuffled']):
            pairs[:,:,ep*2+arm_num]=[[sgn*(int(observations[scene,s,w]['ordered_'+metric])-int(observations[scene,s,w][arm+'_'+metric])) for w in maps] for s in seeds]
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=4000);p.add_argument('--output',default='paired-power.json');opts=p.parse_args()
    rep=opts.replicates;start=monotonic();pilot=load_pilot();rng=np.random.default_rng(RNG_SEED)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    gen=torch.Generator(device='cuda').manual_seed(RNG_SEED+1)
    a=torch.tensor(pilot,device='cuda'); flat=a.reshape(8,-1)
    center=pilot.mean(axis=(0,1),dtype=np.float64)
    targets=np.array([targets for _,_,_,targets in ENDPOINTS],dtype=np.float64)
    shifts=targets[:,:,None]-center.reshape(5,1,2)
    hits=np.zeros((5,3,3),np.int64)
    # The same seed and map draw is used for all arms and endpoints at both levels.
    # Collapsing repeated pilot seed/map identities before multiplication is exact.
    batch=4000
    ns_ones=torch.ones((batch,NS),device='cuda');nw_ones=torch.ones((batch,NW),device='cuda')
    one_verified=False
    for outer in range(rep):
        s=rng.integers(0,8,NS);w=rng.integers(0,64,NW)
        sv=torch.as_tensor(s,device='cuda');wv=torch.as_tensor(w,device='cuda')
        sample_means=pilot[s[:,None],w[None,:]].mean(axis=1,dtype=np.float64).reshape(NS,5,1,2)
        samples=torch.empty((DRAWS,10),device='cuda')
        for offset in range(0,DRAWS,batch):
            si=torch.randint(NS,(batch,NS),device='cuda',generator=gen);wi=torch.randint(NW,(batch,NW),device='cuda',generator=gen)
            c_s=torch.zeros((batch,8),device='cuda').scatter_add_(1,sv[si],ns_ones)
            c_w=torch.zeros((batch,64),device='cuda').scatter_add_(1,wv[wi],nw_ones)
            draw=((c_s@flat).reshape(batch,64,10)*c_w[:,:,None]).sum(dim=1)/(NS*NW)
            samples[offset:offset+batch]=draw
            if not one_verified:
                ii=si[:20].cpu().numpy();jj=wi[:20].cpu().numpy(); truth=pilot[s[ii][:,:,None],w[jj][:,None,:]].mean(axis=(1,2),dtype=np.float64)
                assert np.array_equal(draw[:20].cpu().numpy(),truth);one_verified=True
        ordered=torch.sort(samples,dim=0).values
        bounds=[]
        for ep in range(5):
            q=(.05/6 if ep<3 else .05/4)/2
            z=(DRAWS-1)*q;k=int(z);t=z-k
            get=ordered[k:k+2,2*ep:2*ep+2].to(torch.float64)
            bounds.append(get[0]+t*(get[1]-get[0]))
        lo=torch.stack(bounds).cpu().numpy()[:,None,:]
        favorable=(sample_means+shifts[None,:,:,:]>0).sum(axis=0)
        ok=(lo+shifts>0)&(favorable>=24)
        hits[:,:,:2]+=ok;hits[:,:,2]+=np.all(ok,axis=2)
        if (outer+1)%500==0: print(outer+1,'of',rep,'seconds',round(monotonic()-start,1),flush=True)
    rows=[]
    for idx,(scene,metric,sgn,values) in enumerate(ENDPOINTS):
        for ti,target in enumerate(values):
            n=hits[idx,ti];pr=n/rep
            rows.append(dict(scene=scene,metric=metric,target_gain=target,masked=float(pr[0]),shuffled=float(pr[1]),joint=float(pr[2]),joint_successes=int(n[2]),joint_mcse=float(np.sqrt(pr[2]*(1-pr[2])/rep))))
    record=dict(status='post-run methodological check using pilot data only',pilot_sha256=PIN,pilot_shape=[8,64],future_shape=[NS,NW],outer_replicates=rep,inner_draws_per_replicate=DRAWS,outer_rng_seed=RNG_SEED,inner_rng_seed=RNG_SEED+1,minimum_favorable_training_seeds=24,source='empirical centered paired contrast vectors plus target location shifts, including fraction differences; does not generate individual Bernoulli outcomes',common_seed_map_draws_across_contrasts=True,cpu_crosscheck_20_inner_draws=one_verified,script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),seconds=round(monotonic()-start,2),results=rows)
    dest=HERE/opts.output;tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(dest)
    print('complete',rep,'outer replicates, elapsed',record['seconds'],flush=True)

if __name__=='__main__':main()
