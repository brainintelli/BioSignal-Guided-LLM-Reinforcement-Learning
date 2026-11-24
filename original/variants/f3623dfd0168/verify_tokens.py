"""Rebuild actual input IDs with the pinned tokenizer, without loading weights."""
import json,sys
from pathlib import Path
root=Path('/home/researcher/meadowmind-gpu/m8b-project');sys.path.insert(0,str(root))
from transformers import AutoTokenizer
from agent.m8x_contract import encode,target_messages
from agent.m8x_causal_kv import normalize_target
folder=Path('/home/researcher/meadowmind-gpu/diagnostics/rl_kv2_registered_20261107')
reg=json.loads(Path('/home/researcher/meadowmind-gpu/m8x-kv-runs/kv_attempt2/registration.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(reg['config']['model'],local_files_only=True)
stub=tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
count=targets=0
for mode in reg['config']['performance']['arms']:
    for line in (folder/(mode+'_transitions.jsonl')).read_text().splitlines():
        row=json.loads(line);prepared=encode(row['observation'])
        text=tokenizer.apply_chat_template(prepared['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert ids==row['executed_verb_token_ids']
        target=None;v=row['verb_index']
        if len(prepared['branches'][v])>1:
            text=tokenizer.apply_chat_template(target_messages(prepared,v),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            target=tokenizer.encode(text,add_special_tokens=False)
            if mode=='optimized':target=normalize_target(ids,target,len(ids)-len(stub),reg['config']['max_tokens'])
            targets+=1
        assert target==row['executed_target_token_ids'];count+=1
result=dict(passed=True,verb_inputs_verified=count,target_inputs_verified=targets,
            normalized_target_template_explicit=True,empty_thinking_stub_tokens=len(stub),model_weights_loaded=False)
(folder/'token_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
