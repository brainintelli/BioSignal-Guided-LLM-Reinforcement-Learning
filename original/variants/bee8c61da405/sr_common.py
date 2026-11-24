"""Authenticated runtime helpers for the isolated five-arm experiment."""
import gc,hashlib,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import torch
import base_smoke as b
from sr_data import Data,sha,read,write,digest,canonical
from sr_policy import Policy,load
from sr_protocol import PROTOCOL,STATE_ARMS,REWARD_ARMS,ARMS
from host_cache import release_host_cache
ROOT=Path(__file__).resolve().parent;b.WORK=ROOT;b.PROJECT=ROOT/'project'
PIN='c06cfa4acdbafd856a4ff3929c5847657f96c68e2d38c74f9c1c6df527ef2000'
SOURCE_FILES=['sr_protocol.py','sr_data.py','sr_contract.py','sr_policy.py','sr_rollout.py','sr_common.py','sr_run.py','sr_verify.py','host_cache.py','batch_boundary.py']

def config(arm,block):
    if arm not in ARMS or not 0<=block<8:raise ValueError('configuration identity')
    c={k:v for k,v in b.CONFIG.items() if k not in ('alpha','rng_seed','r2_enabled','state2_enabled')}
    return c|dict(rng_seed=PROTOCOL['initial_seed']+block,r1_weight=1.,aux_weight=.05 if arm in REWARD_ARMS else 0.,
        reward_contract='additive_experimental_v1',r2_enabled=arm in REWARD_ARMS,state2_enabled=arm in STATE_ARMS,r2_biologically_validated=False)

def source_identity():return dict(base=b.source_identity(),files={n:sha(ROOT/n) for n in SOURCE_FILES},dataset_sha256=sha(ROOT/'dataset.json'),protocol=PROTOCOL)

def data():return Data(ROOT/'dataset.json',PIN)

def policy(arm,block):
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    if torch.__version__!='2.6.0+cu124' or torch.cuda.device_count()!=1:raise ValueError('authenticated runtime and one visible GPU per seedblock required')
    p=Policy(SimpleNamespace(**config(arm,block)),arm,PIN)
    if any('lora_' not in n for n,v in p.model.named_parameters() if v.requires_grad):raise ValueError('non-LoRA trainable parameter')
    p.assert_reference_frozen();return p

def state(p):return dict(actor=p.digest(),critic=b.tensor_hash(p.critic.named_parameters()),base=b.tensor_hash((n,v) for n,v in p.model.named_parameters() if not v.requires_grad),reference=b.tensor_hash(p.reference_model.named_parameters()))

def optimizers(p):
    c=p.c
    return dict(actor=torch.optim.AdamW([v for v in p.model.parameters() if v.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0),
        critic=torch.optim.AdamW(p.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0))

def steps(opts):return {k:[int(s['step'].item()) for s in o.state.values() if 'step' in s] for k,o in opts.items()}

def tensor_hash(items):
    h=hashlib.sha256()
    for name,t in items:
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def optimizer_hash(opts):
    entries=[]
    for name,opt in sorted(opts.items()):
        for i,s in enumerate(opt.state.values()):
            entries.extend((f'{name}/{i}/{k}',v) for k,v in sorted(s.items()))
    return tensor_hash(entries)

def disk_optimizer(path):
    value=torch.load(path,map_location='cpu',weights_only=False);items=[];counts={}
    for name in ('actor','critic'):
        counts[name]=[int(s['step'].item()) for s in value[name]['state'].values() if 'step' in s]
        for i,s in enumerate(value[name]['state'].values()):items.extend((f'{name}/{i}/{k}',v) for k,v in sorted(s.items()))
    return tensor_hash(items),counts

def stats(p,rows):return b.statistics(p,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])

def environment():return b.GodotEnv(str(b.GODOT),str(b.PROJECT),map_mode=b.CONFIG['map_mode'],navigation_steps=b.CONFIG['navigation_steps'])

def resource(p,release=True):
    p.clear_prefix_cache();gc.collect();torch.cuda.synchronize()
    def status(k):return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith(k+':')))*1024
    before=status('VmRSS')
    if release:release_host_cache()
    return dict(rss_before_release=before,rss_after_release=status('VmRSS'),process_peak_rss=status('VmHWM'),allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())

def device():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible.isdecimal() and 0<=int(visible)<8
    physical=int(visible);lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines();row=lines[physical].split(',')
    return dict(physical_gpu=physical,uuid=row[1].strip(),name=row[2].strip(),logical_gpu=torch.cuda.current_device(),pid=os.getpid())

def manifest(out,name):write(out/name,{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})

def check_manifest(out,name,extras=()):
    m=read(out/name);files={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}-{name,*extras}
    if files!=set(m):raise ValueError('artifact inventory changed')
    for n,h in m.items():
        p=(out/n).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p)!=h:raise ValueError('artifact changed '+n)

def emit(event,**fields):print(canonical(dict(event=event,time=time.time(),**fields)),flush=True)
