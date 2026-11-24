"""Export actual recorded facts for a tokenizer-only measurement on node 5."""
import hashlib
import json
from pathlib import Path
from semantic_candidate import build, render_concise, CONCISE_RULES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
trace = ROOT / 'output/rl_kv2_20261107/optimized_transitions.jsonl'
rows = [json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines()]
records = []
for i,row in enumerate(rows):
    variants = {mode:build(row,mode) for mode in ('grid','sparse')}
    variants['concise'] = {**variants['sparse'], 'messages':[
        {'role':'system','content':CONCISE_RULES},
        {'role':'user','content':render_concise(variants['sparse']['state'])}]}
    actions = variants['sparse']['action_catalog']
    selected = next(a for a in actions if a['engine_action'] == row['action'])
    records.append({'index':i,'seed':row['seed'],'update':row['update'],'step':row['step'],'env':row['env'],
                    'baseline':row['model_input'],'baseline_expected_ids':row['executed_verb_token_ids'],
                    'variants':{k:v['messages'] for k,v in variants.items()},
                    'selected':selected,'catalog':actions})
    if i == 0:
        (HERE/'real_candidate.json').write_text(json.dumps({
            'source_trace':str(trace.relative_to(ROOT)), 'seed':row['seed'], 'update':row['update'],
            'step':row['step'], 'env':row['env'], 'real_original_action':row['action'],
            'semantic_relabel_of_same_action':selected, 'real_original_reward':row['reward'],
            'notice':'Semantic text is a deterministic relabel, not a newly sampled model action.',
            **variants['concise']},ensure_ascii=False,indent=2),encoding='utf-8')
        (HERE/'real_state.txt').write_text(variants['concise']['messages'][1]['content'],encoding='utf-8')
(HERE/'probe_inputs.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
reg = json.loads((ROOT/'output/rl_kv2_20261107/registration.json').read_text(encoding='utf-8'))
mismatches = [p for p,h in reg['source_hashes'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
receipt = {'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'records':len(rows),
           'total_legal_actions':sum(len(r['catalog']) for r in records),
           'frozen_sources_checked':len(reg['source_hashes']),'frozen_source_mismatches':mismatches,
           'formal_training_started':False, 'weights_loaded':False}
(HERE/'local_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
assert not mismatches
