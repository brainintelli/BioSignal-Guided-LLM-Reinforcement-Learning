"""Diagnostic semantic PPO policy with exact finite-trie action probabilities.

No free-form generation. A sampled leaf denotes the entire semantic target;
unary edges are forced by the legal-choice grammar. All branching logits are
computed, so entropy/KL cover the complete conditional target distribution.
"""
from contextlib import nullcontext
import torch
from transformers.cache_utils import DynamicCache,LinearAttentionLayer
from agent.m8x_fast_policy import FunctionalLinearLayer
from agent.m8x_grouped_lora import GroupedPolicy
from agent.m8x_kv_policy import branch_cache
from agent.m8x_causal_kv import normalize_target
from agent.m8s_contract import encode,target_messages,CODES,VERSION
from agent.m8s_trie import choice_tree


class SemanticPolicy(GroupedPolicy):
    def __init__(self,*args,**kwargs):
        # Verify and load the old checkpoint before changing its interface.
        super().__init__(*args,**kwargs)
        self.verb_ids=self._single_ids(CODES)
        self.direction_ids=self._single_ids(('向上','向右','向下','向左'))
        self.target_ids=[]
        self.semantic_node_calls=0

    def prepare(self,observations,salt='m6'):
        prepared=[]
        stub=self.tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
        for obs in observations:
            row=encode(obs,salt)
            ids=self._tokens(row['messages'])
            if ids[-len(stub):]!=stub:raise ValueError('Unexpected thinking stub')
            row.update(input_ids=ids,kv_prefix_ids=ids,assistant_boundary=len(ids)-len(stub),
                       normalized_targets={},semantic_trees={})
            for v,branch in enumerate(row['branches']):
                if len(branch)>1:
                    sequences=[self.tokenizer.encode(x,add_special_tokens=False) for x in row['target_texts'][v]]
                    row['semantic_trees'][v]=choice_tree(sequences,self.tokenizer.eos_token_id)
            prepared.append(row)
        return prepared

    def _target_tokens(self,row,verb):
        if verb not in row['normalized_targets']:
            original=self._tokens(target_messages(row,verb))
            row['normalized_targets'][verb]=normalize_target(row['input_ids'],original,row['assistant_boundary'],self.max_tokens)
        return row['normalized_targets'][verb]

    def _semantic_logits(self,row,ids,token_ids,reference):
        scope=self._kv_scope
        if scope is None or scope['reference']!=reference:raise RuntimeError('No semantic graph scope')
        if len(ids)>self.max_tokens:raise ValueError('Semantic context overflow; no truncation')
        model=self.reference_model if reference else self.model
        prefix=tuple(row['kv_prefix_ids'])
        nodes=scope.setdefault('semantic_nodes',{})
        if scope['roots'][prefix] is None:
            # This also enforces the same prefill shape for sampling and training.
            self._logits([list(prefix)],self.verb_ids,reference)
        nodes.setdefault(prefix,scope['roots'][prefix])
        key=tuple(ids)
        if key not in nodes:
            ancestors=[p for p in nodes if len(p)<len(key) and key[:len(p)]==p and p[:len(prefix)]==prefix]
            ancestor=max(ancestors,key=len)
            cache,last=nodes[ancestor]
            copied=branch_cache(cache)
            out=self._backbone(model,list(key[len(ancestor):]),copied)
            nodes[key]=(copied,out.last_hidden_state[:,-1,:])
            self.semantic_node_calls+=1
        last=nodes[key][1]
        weights=model.get_output_embeddings().weight[token_ids].float()
        return torch.nn.functional.linear(last.float(),weights)[0]

    def target_distribution(self,prepared,verbs,reference=False):
        with self._scope(prepared,reference), (torch.no_grad() if reference else nullcontext()):
            rows=[]
            for row,verb in zip(prepared,map(int,verbs)):
                size=len(row['branches'][verb])
                if size<1:raise ValueError('Illegal semantic verb')
                if size==1:
                    logps=torch.zeros(1,device=self.device_name)
                else:
                    prompt=self._target_tokens(row,verb)
                    nodes,paths=row['semantic_trees'][verb]
                    decisions={}
                    # Canonical prefix/token order makes branch calculation independent
                    # of leaf ordering and external minibatch grouping.
                    for prefix,tokens in nodes.items():
                        logits=self._semantic_logits(row,prompt+list(prefix),tokens,reference)
                        decisions[prefix]=torch.log_softmax(logits,dim=-1)
                    logps=torch.stack([sum((decisions[p][i] for p,i in path),
                                          torch.zeros((),device=self.device_name)) for path in paths])
                rows.append(torch.nn.functional.pad(logps,(0,64-size),value=-1.e9))
            return torch.distributions.Categorical(logits=torch.stack(rows))

    def metadata(self):
        result=super().metadata()
        result.update(contract=VERSION,inference_layout='semantic_unpadded_grouped_trie_v1',
                      target_token_ids='dynamic_complete_semantic_labels',direction_token_ids=self.direction_ids,
                      action_probability='P(semantic verb) * product(masked branching-token probabilities)',
                      target_sampling='complete finite-trie leaf; unary edges forced; explicit EOS',
                      reference_interface='same semantic state and action grammar; frozen parent weights',
                      formal_training_started=False,release_authorized=False)
        return result
