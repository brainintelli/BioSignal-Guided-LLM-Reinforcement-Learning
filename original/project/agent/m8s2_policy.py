"""Canonical per-state branch batches and versioned static-prefix KV reuse."""
from contextlib import nullcontext
import torch
from agent.m8s_policy import SemanticPolicy
from agent.m8s2_contract import encode,CODES,RULES,VERSION
from agent.m8s2_cache import fresh_cache,repeat_cache
from agent.m8s2_kernels import pin_kernels
from agent.m8s_trie import choice_tree
from agent.m8x_kv_policy import branch_cache

class BatchedSemanticPolicy(SemanticPolicy):
    def __init__(self,*args,**kwargs):
        self.pinned_kernels=pin_kernels()
        super().__init__(*args,**kwargs)
        text=self.tokenizer.apply_chat_template([{'role':'system','content':RULES},{'role':'user','content':'boundary'}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        self.system_ids=self.tokenizer.encode(text.split('<|im_start|>user\n')[0],add_special_tokens=False)
        self._persistent_prefixes={}
        self.cache_stats=dict(system_hits=0,system_misses=0,state_hits=0,state_misses=0,branch_batches=0,branch_rows=0,forward_tokens=0,forward_calls=0)

    def clear_prefix_cache(self):self._persistent_prefixes.clear()

    def prepare(self,observations,salt='m6'):
        result=[]
        for obs in observations:
            row=encode(obs,salt);ids=self._tokens(row['messages'])
            row.update(input_ids=ids,kv_prefix_ids=ids,normalized_targets={},semantic_trees={})
            for v,branch in enumerate(row['branches']):
                if len(branch)>1:
                    sequences=[self.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
                    row['semantic_trees'][v]=choice_tree(sequences,self.tokenizer.eos_token_id)
            result.append(row)
        return result

    def _target_tokens(self,row,verb):
        # Structured semantic heads share the same state representation. The
        # selected verb conditions the legal target set, not another LM pass.
        return row['input_ids']

    def _forward_batch(self,model,sequences,cache):
        start=cache.get_seq_length();lengths=[len(s) for s in sequences]
        if not lengths or min(lengths)<1:raise ValueError('Empty semantic continuation')
        width=max(2,max(lengths));batch=len(lengths)
        if start+width>self.max_tokens:raise ValueError('Context overflow; never truncate')
        ids=torch.tensor([s+[self.tokenizer.pad_token_id]*(width-len(s)) for s in sequences],device=self.device_name)
        mask=torch.tensor([[1]*(start+n)+[0]*(width-n) for n in lengths],device=self.device_name)
        pos=torch.arange(start,start+width,device=self.device_name).view(1,1,-1).expand(3,batch,-1)
        out=model.get_base_model().model(input_ids=ids,position_ids=pos,attention_mask=mask,past_key_values=cache,use_cache=True,return_dict=True)
        self.cache_stats['forward_calls']+=1;self.cache_stats['forward_tokens']+=batch*width
        last=out.last_hidden_state[torch.arange(batch,device=self.device_name),torch.tensor(lengths,device=self.device_name)-1,:]
        return last

    def _root(self,ids,reference):
        scope=self._kv_scope;key=tuple(ids)
        if key not in scope['roots']:raise ValueError('Unregistered state root')
        if scope['roots'][key] is not None:
            self.cache_stats['state_hits']+=1;return scope['roots'][key]
        self.cache_stats['state_misses']+=1
        model=self.reference_model if reference else self.model
        if ids[:len(self.system_ids)]==self.system_ids:
            static=scope.get('static_prefix')
            if static is None:
                revision=self.execution_cache.revisions[reference]
                existing=self._persistent_prefixes.get(reference) if not torch.is_grad_enabled() else None
                if existing is not None and existing[0]==revision:
                    static=existing[1];self.cache_stats['system_hits']+=1
                else:
                    static=fresh_cache(model.get_base_model().config)
                    self._forward_batch(model,[self.system_ids],static)
                    self.cache_stats['system_misses']+=1
                    if not torch.is_grad_enabled():self._persistent_prefixes[reference]=(revision,static)
                scope['static_prefix']=static
            else:self.cache_stats['system_hits']+=1
            cache=branch_cache(static);tail=ids[len(self.system_ids):]
        else:
            cache=fresh_cache(model.get_base_model().config);tail=ids
        last=self._forward_batch(model,[tail],cache)
        scope['roots'][key]=(cache,last)
        return cache,last

    def _logits(self,inputs,token_ids,reference=False):
        model=self.reference_model if reference else self.model
        weights=model.get_output_embeddings().weight[token_ids].float();answers=[]
        with torch.no_grad() if reference else nullcontext():
            for ids in inputs:
                roots=[p for p in self._kv_scope['roots'] if ids[:len(p)]==list(p)]
                if len(roots)!=1:raise ValueError('Ambiguous state prefix')
                root=roots[0];cache,last=self._root(list(root),reference)
                if len(ids)>len(root):last=self._forward_batch(model,[ids[len(root):]],branch_cache(cache))
                answers.append(torch.nn.functional.linear(last.float(),weights))
        return torch.cat(answers)

    def target_distribution(self,prepared,verbs,reference=False):
        with self._scope(prepared,reference), (torch.no_grad() if reference else nullcontext()):
            model=self.reference_model if reference else self.model;answers=[]
            for row,verb in zip(prepared,map(int,verbs)):
                size=len(row['branches'][verb])
                if size<1:raise ValueError('Illegal semantic verb')
                if size==1:logps=torch.zeros(1,device=self.device_name)
                else:
                    target_cache,last=self._root(row['input_ids'],reference);first=last[0]
                    nodes,paths=row['semantic_trees'][verb];prefixes=[prefix for prefix in nodes if prefix]
                    hidden={():first}
                    if prefixes:
                        branch=repeat_cache(target_cache,len(prefixes))
                        rows=self._forward_batch(model,[list(s) for s in prefixes],branch)
                        hidden.update(zip(prefixes,rows));self.cache_stats['branch_batches']+=1;self.cache_stats['branch_rows']+=len(prefixes)
                    decisions={}
                    for prefix,tokens in nodes.items():
                        weights=model.get_output_embeddings().weight[tokens].float()
                        logits=torch.nn.functional.linear(hidden[prefix].float(),weights)
                        decisions[prefix]=torch.log_softmax(logits,dim=-1)
                    logps=torch.stack([sum((decisions[p][i] for p,i in path),torch.zeros((),device=self.device_name)) for path in paths])
                answers.append(torch.nn.functional.pad(logps,(0,64-size),value=-1.e9))
            return torch.distributions.Categorical(logits=torch.stack(answers))

    def metadata(self):
        result=super().metadata()
        result.update(contract=VERSION,inference_layout='semantic_v2_static_prefix_batched_branches',
            pinned_fla_configs=self.pinned_kernels,target_template='shared_state_semantic_heads',
            action_probability='P(semantic verb|state) * constrained semantic-target trie conditioned on the verb legal set; shared root context',
            cache_lifetime='static prefix: versioned no-grad or graph-local; state: graph-local',
            branch_layout='one canonical batch of all nonroot decision prefixes per state; short right padding only')
        return result
