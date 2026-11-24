"""Independent engine/token replay and fresh-checkpoint logp audit of diagnostics."""
import gc,json,math,statistics,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from agent.m8x_policy import Policy
from agent.m8x_grouped_lora import GroupedPolicy
from agent.m8s_policy import SemanticPolicy
from agent import m8x_contract as old,m8s_contract as new
from agent.m8x_causal_kv import normalize_target
from agent.m8s_trie import choice_tree
from agent.m8x_env import GodotEnv
from tools.audit_m6_run import assert_equivalent
from tools.benchmark_m8s import sha,verify,write

warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*',module=r'fla\.ops\.gated_delta_rule\.chunk')
torch.set_num_threads(8)
reg=json.loads(Path(sys.argv[1]).read_text());base=Path(sys.argv[2]);c=reg['config'];verify(reg)
tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
def tokens(messages):
    return tokenizer.encode(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
stub=tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
records={mode:[json.loads(x) for x in (base/(mode+'_transitions.jsonl')).read_text().splitlines()] for mode in c['arms']}
summary={}
common=[r['observation'] for r in records['baseline'] if r['update']==1][:4]
for mode in ['old_optimized']:
    cls={'baseline':Policy,'old_optimized':GroupedPolicy,'semantic':SemanticPolicy}[mode]
    p=cls(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=True,seed=c['rng_seed'],
          **({} if mode=='baseline' else {'registration':reg}))
    initial=p.digest();first=[r for r in records[mode] if r['update']==1];errors=[]
    with torch.no_grad():
        for start in range(0,len(first),4):
            batch=first[start:start+4];prepared=p.prepare([r['observation'] for r in batch]);verbs=[r['verb_index'] for r in batch]
            v=torch.tensor(verbs,device='cuda');t=torch.tensor([r['target_index'] for r in batch],device='cuda')
            vd,td=p.distributions(prepared,verbs);values=(vd.log_prob(v)+td.log_prob(t)).tolist()
            errors.extend(abs(value-r['log_probability']) for value,r in zip(values,batch))
    profile=[]
    if mode!='baseline':
        prepared=p.prepare(common)
        with torch.no_grad():
            for repeat in range(4):
                for row in prepared:
                    if len(row['branches'][1])<2:continue
                    with p._scope([row],False):
                        torch.cuda.synchronize();start=time.perf_counter();p.verb_distribution([row]);torch.cuda.synchronize();middle=time.perf_counter()
                        p.target_distribution([row],[1]);torch.cuda.synchronize();end=time.perf_counter()
                    if repeat:profile.append({'verb_seconds':middle-start,'navigation_target_seconds':end-middle})
    summary[mode]={'first_round_all_logps_checked':len(errors),'first_round_logp_error_max':max(errors),'weights_unchanged':p.digest()==initial,
                   'component_profile_samples':len(profile),
                   'component_profile':None if not profile else {k:statistics.median(x[k] for x in profile) for k in profile[0]}}
    print(json.dumps({'pre_assert':mode,'max_error':max(errors),'first_errors':errors[:8],'weights_unchanged':summary[mode]['weights_unchanged']}),flush=True)
    assert len(errors)==128 and summary[mode]['weights_unchanged']
    print(json.dumps({'event':'fresh_checkpoint_logps','mode':mode,**summary[mode]}),flush=True)
    del p,prepared,vd,td;gc.collect();torch.cuda.empty_cache()

write(base/'reload_probe_with_reference.json',summary)
