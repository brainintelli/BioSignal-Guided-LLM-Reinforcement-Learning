"""CPU-only, local-files-only tokenizer audit. No model weights or rollout."""
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
from transformers import AutoTokenizer

source, output, model = map(Path,sys.argv[1:4])
tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)

def ids(messages):
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return tokenizer.encode(text,add_special_tokens=False)

def stats(v):
    v=sorted(v)
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),
                p95=v[min(len(v)-1,int(.95*len(v)))],max=max(v),over_1536=sum(x>1536 for x in v))

records=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
words=['A','B','move','go','take','eat','drink','rest','camp','wait',
       '向上','向右','向下','向左','向前','向后','移动','前往','采集','吃饭','喝水','休息','建造','等待',
       '向上走一格','建营','树木（左3格、上1格）']
vocab=[dict(text=w,ids=tokenizer.encode(w,add_special_tokens=False)) for w in words]
lengths={key:[] for key in ['baseline','grid','sparse','concise']}
states={key:[] for key in lengths}
target_lengths=[]
action_lengths=[]
prefix_errors=[]
baseline_errors=[]
for r in records:
    all_messages={'baseline':r['baseline'],**r['variants']}
    for key,m in all_messages.items():
        token_ids=ids(m)
        lengths[key].append(len(token_ids))
        states[key].append(len(tokenizer.encode(m[1]['content'],add_special_tokens=False)))
        if key=='baseline' and token_ids!=r['baseline_expected_ids']:
            baseline_errors.append(r['index'])
    # Verify single-token semantic verbs/directions at the real candidate chat boundary.
    prefix=tokenizer.apply_chat_template(r['variants']['sparse'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix_ids=tokenizer.encode(prefix,add_special_tokens=False)
    for w in words[10:24]:
        together=tokenizer.encode(prefix+w,add_special_tokens=False)
        if together!=prefix_ids+tokenizer.encode(w,add_special_tokens=False):
            prefix_errors.append([r['index'],w])
    for action in r['catalog']:
        action_lengths.append(len(tokenizer.encode(action['text'],add_special_tokens=False)))
        if action['engine_action']['verb'] in ('navigate','gather'):
            target_lengths.append(len(tokenizer.encode(action['target'],add_special_tokens=False)))
result={'hostname':platform.node(),'device':'CPU; tokenizer only; no model weights loaded',
        'tokenizer_path':str(model), 'tokenizer_file_sha256':{
          f:hashlib.sha256((model/f).read_bytes()).hexdigest() for f in ('tokenizer.json','tokenizer_config.json','chat_template.jinja')},
        'probe_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows':len(records),'vocabulary':vocab,
        'verb_prompt_tokens':{k:stats(v) for k,v in lengths.items()},
        'state_only_tokens':{k:stats(v) for k,v in states.items()},
        'semantic_target_tokens':stats(target_lengths),'full_action_text_tokens':stats(action_lengths),
        'baseline_token_mismatches':baseline_errors,'semantic_boundary_mismatches':prefix_errors,
        'first_row':{k:v[0] for k,v in lengths.items()},
        'limitations':'Lengths are not latency or understanding scores. Recorded states cover ticks 0..105, not night.'}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
assert not baseline_errors and not prefix_errors
