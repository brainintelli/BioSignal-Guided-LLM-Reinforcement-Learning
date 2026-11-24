"""M9 online runtime. Importing this module loads no model, SDK or socket.

Only the rank-zero runtime owns Godot. Replica workers consume frozen PPO
inputs; they cannot acquire biological observations or sample game actions.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

CONTRACT = 'mea_online_transport_v1'
EXPECTED_ACTOR = '94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
FEATURE_NAMES = ('count', 'rate_per_channel', 'active_fraction', 'mean_abs_amplitude_uv')
SNAPSHOT_KEYS = frozenset(('schema', 'snapshot_id', 'binding_sha256', 'request_id',
    'window_sha256', 'source', 'context_id', 'start_frame', 'end_frame', 'feature_values',
    'normalized_values', 'normalization_id', 'coverage_basis'))
DEFAULTS = dict(map_mode='infinite_freeze', navigation_steps=4, state_variant='reserves',
    reward_variant='bottleneck', gamma_tick=.99985, gae_lambda=.999, actor_lr=2e-6,
    critic_lr=1e-4, epochs=2, critic_epochs=16, minibatch=16, microbatch=2,
    critic_minibatch=16, clip=.1, reference_kl=.02, entropy_coef=.002,
    actor_grad_clip=1., critic_grad_clip=1., target_kl=.015,
    zero_update_logp_tolerance=1e-6, max_tokens=4096, verb_exploration_epsilon=.1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _read_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    def constant(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=constant)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite number required: ' + label)
    return value


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA256 required: ' + label)


def validate_runtime_config(config):
    if config.get('r2_enabled', False) is not False or config.get('stimulation', False) is not False:
        raise ValueError('R2 and stimulation must be disabled')
    if config.get('alpha', 1.) != 1.:
        raise ValueError('R1 coefficient must remain exactly one')
    if config.get('source') not in ('live', 'simulation'):
        raise ValueError('explicit source required')
    _hash(config.get('binding_sha256'), 'binding')
    if not isinstance(config.get('normalization_id'), str) or not config['normalization_id']:
        raise ValueError('normalization identity required')
    devices = config.get('devices')
    if (not isinstance(devices, list) or not devices or len(set(devices)) != len(devices)
        or any(type(d) is not int or d < 0 for d in devices) or len(devices) not in (1, 8)):
        raise ValueError('one device or eight distinct devices required')
    if type(config.get('batch_size')) is not int or not 1 <= config['batch_size'] <= 128:
        raise ValueError('batch size must be in 1..128')
    paths = config.get('model_paths', {})
    if set(paths) != {'model', 'initial_checkpoint', 'parent_checkpoint', 'warm_checkpoint'}:
        raise ValueError('four explicit model paths required')
    if any(not isinstance(p, str) or not p for p in paths.values()):
        raise ValueError('model paths must be nonempty strings')
    for key in ('map_mode', 'navigation_steps', 'state_variant', 'reward_variant', 'gamma_tick'):
        if key in config and config[key] != DEFAULTS[key]:
            raise ValueError('warm model identity differs: ' + key)
    return deepcopy(config)


def project_observation(observation, snapshot, config):
    """Strict positive projection; raw identifiers stay outside model messages."""
    from agent.m9_contract import encode
    from agent.m8s3_supply import readiness_features
    if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_KEYS:
        raise ValueError('unexpected online snapshot fields')
    snapshot = json.loads(canonical(snapshot))
    if snapshot['snapshot_id'] != digest({k:v for k,v in snapshot.items() if k != 'snapshot_id'}):
        raise ValueError('snapshot digest differs')
    for key in ('snapshot_id', 'binding_sha256', 'window_sha256', 'context_id'):
        _hash(snapshot[key], key)
    if snapshot['schema'] != 'mea.online.snapshot.v1' or snapshot['coverage_basis'] != 'filtered_transport_frames':
        raise ValueError('online transport coverage required')
    for key in ('source', 'binding_sha256', 'normalization_id'):
        if snapshot[key] != config[key]:
            raise ValueError('snapshot identity differs: ' + key)
    if snapshot['source'] not in ('live', 'simulation') or not isinstance(snapshot['request_id'], str) or not snapshot['request_id']:
        raise ValueError('invalid source/request')
    if any(type(snapshot[k]) is not int or snapshot[k] < 0 for k in ('start_frame', 'end_frame')):
        raise ValueError('integer frame interval required')
    if snapshot['end_frame'] - snapshot['start_frame'] != 20000:
        raise ValueError('one-second transport frame window required')
    for key in ('feature_values', 'normalized_values'):
        if not isinstance(snapshot[key], list) or len(snapshot[key]) != 4:
            raise ValueError('exactly four features required')
        for value in snapshot[key]:_number(value, key)
    features, normalized = snapshot['feature_values'], snapshot['normalized_values']
    if any(v < 0 for v in features) or features[0] != int(features[0]) or features[2] > 1:
        raise ValueError('invalid raw feature range')
    if any(abs(v) > 5 for v in normalized):
        raise ValueError('normalized features outside fixed clip range')
    if observation['map_mode'] != DEFAULTS['map_mode'] or observation['navigation_steps'] != DEFAULTS['navigation_steps']:
        raise ValueError('environment differs from warm policy')
    row = encode(observation, DEFAULTS['state_variant'])
    if digest(row) != snapshot['context_id']:
        raise ValueError('snapshot belongs to a different game observation')
    row['messages'][1]['content'] += '\n湿件观测：' + canonical(dict(zip(FEATURE_NAMES, features)))
    row.update(contract=CONTRACT, base_contract='m9_research_v1', snapshot=snapshot,
               critic_input=row['features'] + readiness_features(row) + normalized)
    return row


def trajectory(rows, bootstrap, gamma, lam):
    """Existing duration-aware GAE, applied to a detached batch."""
    from agent.rl_contract import gae
    if not rows or not 0 < gamma < 1 or not 0 < lam <= 1:
        raise ValueError('invalid trajectory')
    _number(bootstrap, 'bootstrap')
    result = deepcopy(rows)
    for row in result:
        for key in ('reward', 'value'):_number(row[key], key)
        if type(row['duration']) is not int or row['duration'] <= 0:
            raise ValueError('positive native duration required')
        if type(row['terminal']) is not bool or type(row['boundary']) is not bool:
            raise ValueError('boolean terminal/boundary required')
        if row['terminal']:row['boundary'] = True
    if result[-1]['terminal'] and bootstrap != 0.:
        raise ValueError('terminal bootstrap must be zero')
    result[-1]['boundary'] = True
    next_values = [0. if r['terminal'] else result[i+1]['value'] if i+1 < len(result) else bootstrap
                   for i,r in enumerate(result)]
    adv, returns = gae([r['reward'] for r in result], [r['value'] for r in result], next_values,
        [r['duration'] for r in result], [r['terminal'] for r in result], [r['boundary'] for r in result], gamma, lam)
    for row, a, ret in zip(result, adv, returns):row.update(advantage=a, **{'return':ret})
    return result


def expand_online_critic(critic):
    import torch
    old=critic[0]
    if tuple(old.weight.shape) != (128,108):raise ValueError('expected warm 108-D critic')
    new=torch.nn.Linear(112,128,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:108].copy_(old.weight);new.bias.copy_(old.bias)
    critic[0]=new


def critic_values(policy, prepared):
    import torch
    if not prepared or any(r.get('contract') != CONTRACT or len(r['critic_input']) != 112 for r in prepared):
        raise ValueError('online critic input contract')
    facts=torch.tensor([r['critic_input'] for r in prepared],device=policy.device_name,dtype=torch.float32)
    # Use the same B1 FP32 reduction layout during collection and PPO replay.
    return torch.cat([policy.critic(facts[i:i+1]) for i in range(len(prepared))]).squeeze(-1)


def tensor_hash(items):
    import torch
    h=hashlib.sha256()
    for name,tensor in items:
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _materialize(policy, prepared):
    """Restore deterministic tuple-key trie caches from JSON-safe frozen inputs."""
    from agent.m8s_trie import choice_tree
    if prepared.get('prepared_sha256') != digest({k:v for k,v in prepared.items() if k != 'prepared_sha256'}):
        raise ValueError('frozen policy input changed')
    row=deepcopy(prepared);row['semantic_trees']={}
    for v,branch in enumerate(row['branches']):
        if len(branch)>1:
            sequences=[policy.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
            row['semantic_trees'][v]=choice_tree(sequences,policy.tokenizer.eos_token_id)
    return row


def _difference(a,b):
    if isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b):
        return max((_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return max((_difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        _number(a,'statistic');_number(b,'statistic');return abs(a-b)
    raise ValueError('policy statistic shape differs')


def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class _ModelCore:
    """Model/optimizer state shared by the controller and non-sampling replicas."""
    def __init__(self, policy, config, rank=0, world=1):
        import torch
        from .distributed import Collective
        self.policy,self.config=policy,deepcopy(config)
        training=config.get('training',{})
        if set(training)-set(DEFAULTS):raise ValueError('unknown training parameter')
        protected=('map_mode','navigation_steps','state_variant','reward_variant','gamma_tick',
                   'verb_exploration_epsilon','zero_update_logp_tolerance')
        if any(k in training and training[k] != DEFAULTS[k] for k in protected):
            raise ValueError('cannot override frozen warm/statistic identity')
        self.c=SimpleNamespace(**{**DEFAULTS,**training,'rng_seed':config['seed']})
        for k in ('epochs','critic_epochs','minibatch','microbatch','critic_minibatch'):
            if type(getattr(self.c,k)) is not int or getattr(self.c,k) < 1:raise ValueError('positive '+k+' required')
        self.collective=Collective(policy.device_name,rank,world)
        self.optimizers={
            'actor':torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=self.c.actor_lr,eps=1e-5,weight_decay=0),
            'critic':torch.optim.AdamW(policy.critic.parameters(),lr=self.c.critic_lr,eps=1e-5,weight_decay=0)}
        self.initial=self.state();self.last_update=0;self.last_rows=[]
        self.parameter_ids=self._parameter_ids()
        self.collective.identical(self.initial,'initial model')

    def _parameter_ids(self):
        return {k:[id(p) for group in opt.param_groups for p in group['params']] for k,opt in self.optimizers.items()}

    def state(self):
        p=self.policy
        return dict(actor=tensor_hash((n,t) for n,t in p.model.named_parameters() if t.requires_grad),
                    critic=tensor_hash(p.critic.named_parameters()),
                    base=tensor_hash((n,t) for n,t in p.model.named_parameters() if not t.requires_grad),
                    reference=tensor_hash(p.reference_model.named_parameters()))

    def optimizer_identity(self):
        tensors=[];steps={};groups={}
        for name,opt in sorted(self.optimizers.items()):
            steps[name]=[]
            groups[name]=[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups]
            for i,entry in enumerate(opt.state.values()):
                if 'step' in entry:steps[name].append(int(entry['step'].item()))
                for key,tensor in sorted(entry.items()):tensors.append((f'{name}/{i}/{key}',tensor))
        return dict(tensors=tensor_hash(tensors),steps=steps,groups_sha256=digest(groups))

    def statistics(self, rows):
        import torch
        result=[]
        with torch.no_grad():
            for start in range(0,len(rows),self.c.microbatch):
                part=rows[start:start+self.c.microbatch]
                pp=[_materialize(self.policy,r['prepared']) for r in part]
                verbs=torch.tensor([r['verb'] for r in part],device=self.policy.device_name)
                targets=torch.tensor([r['target'] for r in part],device=self.policy.device_name)
                vd,td=self.policy.distributions(pp,verbs.tolist())
                rv,rt=self.policy.distributions(pp,verbs.tolist(),True)
                values=self.policy.values(pp)
                for j in range(len(part)):
                    result.append(dict(logp=float(vd.log_prob(verbs)[j]+td.log_prob(targets)[j]),
                        verb_logp=float(vd.log_prob(verbs)[j]),value=float(values[j]),
                        reference_verb=rv.probs[j].tolist(),reference_target=rt.probs[j].tolist()))
        return result

    def update(self, rows, bootstrap, update_id):
        import torch
        from .distributed import ppo_update, saved_on_cpu_preserve_layout
        if update_id != self.last_update+1:raise ValueError('nonconsecutive update ID')
        if not 1 <= len(rows) <= self.config['batch_size']:raise ValueError('invalid batch length')
        frozen=json.loads(canonical(rows));input_hash=digest(frozen)
        self.collective.identical(dict(input=input_hash,bootstrap=bootstrap,update_id=update_id),'broadcast batch')
        before=self.state();before_opt=self.optimizer_identity()
        self.collective.identical(dict(model=before,optimizers=before_opt),'pre-update state')
        self.collective.require(all(before[k]==self.initial[k] for k in ('base','reference')),
                                'frozen model/reference changed before update')
        self.policy.assert_reference_frozen()
        for row in frozen:
            _materialize(self.policy,row['prepared'])
            if row['prepared']['branches'][row['verb']][row['target']] != row['action']:
                raise ValueError('action differs from frozen branch')
        stats=self.statistics(frozen)
        expected=[{k:r[k] for k in ('logp','verb_logp','value','reference_verb','reference_target')} for r in frozen]
        error=_difference(stats,expected)
        self.collective.require(error <= self.c.zero_update_logp_tolerance,'old policy statistics differ')
        train_rows=trajectory(frozen,bootstrap,self.c.gamma_tick,self.c.gae_lambda)
        for row in train_rows:row['prepared']=_materialize(self.policy,row['prepared'])
        with saved_on_cpu_preserve_layout():
            metrics=ppo_update(self.policy,self.optimizers,train_rows,self.c,self.collective,update_id)
        after=self.state();after_opt=self.optimizer_identity()
        checks=dict(base_unchanged=after['base']==self.initial['base'],
                    reference_unchanged=after['reference']==self.initial['reference'],
                    inputs_unchanged=digest(rows)==input_hash,
                    optimizer_objects_retained=self.parameter_ids==self._parameter_ids(),
                    finite_parameters=all(bool(torch.isfinite(p).all()) for model in (self.policy.model,self.policy.critic) for p in model.parameters()),
                    finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for model in (self.policy.model,self.policy.critic) for p in model.parameters()),
                    no_frozen_gradients=all(p.grad is None for p in self.policy.model.parameters() if not p.requires_grad))
        for name,key in (('actor','optimizer_steps'),('critic','critic_optimizer_steps')):
            previous=before_opt['steps'][name] or [0]*len(after_opt['steps'][name])
            checks[name+'_optimizer_continuity']=len(previous)==len(after_opt['steps'][name]) and all(
                new-old==metrics[key] for old,new in zip(previous,after_opt['steps'][name]))
        self.collective.require(all(checks.values()),'post-update invariant failed: '+canonical(checks))
        self.collective.identical(dict(model=after,optimizers=after_opt,metrics=metrics),'post-update state')
        self.last_update=update_id;self.last_rows=frozen
        return {**metrics,'old_statistics_max_error':error,'checks':checks,
                'actor_changed':after['actor']!=before['actor'],'critic_changed':after['critic']!=before['critic'],
                'before':before,'after':after,'optimizer_before':before_opt,'optimizer_after':after_opt,
                'input_sha256':input_hash,'r2_enabled':False,'stimulation':False,'reward':'native_R1'}

    def checkpoint(self, path, registration_sha, update_id):
        import torch
        _hash(registration_sha,'registration')
        if update_id != self.last_update:raise ValueError('checkpoint update differs')
        path=Path(path);staging=path.with_name(path.name+'.partial')
        state=self.state();identity=self.optimizer_identity();stats=self.statistics(self.last_rows)
        if path.exists() or staging.exists():raise FileExistsError('never overwrite checkpoint')
        staging.mkdir(parents=True,exist_ok=False)
        torch.save({n:p.detach().cpu() for n,p in self.policy.model.named_parameters() if p.requires_grad},staging/'actor.pt')
        torch.save(self.policy.critic.state_dict(),staging/'critic.pt')
        torch.save({**{k:opt.state_dict() for k,opt in self.optimizers.items()},
                    'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state() if str(self.policy.device_name).startswith('cuda') else None},staging/'optimizers.pt')
        meta=dict(contract=CONTRACT,policy=self.policy.metadata(),config=self.config,update=update_id,
                  registration_sha256=registration_sha,state=state,optimizer=identity,
                  resume_semantics='optimizer continuation only; biological session/episode resume prohibited')
        (staging/'policy.json').write_text(canonical(meta),encoding='utf-8')
        manifest=dict(files={p.name:_file_hash(p) for p in staging.iterdir()},registration_sha256=registration_sha,update=update_id)
        (staging/'manifest.json').write_text(canonical(manifest),encoding='utf-8')
        staging.rename(path)
        self.load_checkpoint(path,registration_sha)
        error=_difference(stats,self.statistics(self.last_rows))
        if self.state()!=state or self.optimizer_identity()!=identity or error>1e-6:
            raise RuntimeError('checkpoint roundtrip mismatch')
        return dict(path=str(path),manifest_sha256=_file_hash(path/'manifest.json'),state=state,
                    roundtrip_verified=True,reload_verified=True,optimizer_roundtrip_verified=True,statistics_max_error=error,
                    update=update_id,registration_sha256=registration_sha)

    def load_checkpoint(self,path,registration_sha):
        import torch
        path=Path(path);manifest=_read_json(path/'manifest.json')
        if manifest['registration_sha256']!=registration_sha:raise ValueError('checkpoint registration differs')
        if set(manifest['files']) != {'actor.pt','critic.pt','optimizers.pt','policy.json'}:
            raise ValueError('checkpoint file set differs')
        for name,value in manifest['files'].items():
            if _file_hash(path/name)!=value:raise ValueError('checkpoint file changed: '+name)
        meta=_read_json(path/'policy.json')
        if meta['contract']!=CONTRACT or meta['config']!=self.config or meta['registration_sha256']!=registration_sha:
            raise ValueError('checkpoint identity differs')
        current=self.state()
        if any(current[k]!=meta['state'][k] for k in ('base','reference')):
            raise ValueError('checkpoint frozen weights differ')
        weights=torch.load(path/'actor.pt',map_location=self.policy.device_name,weights_only=True)
        params={n:p for n,p in self.policy.model.named_parameters() if p.requires_grad}
        if set(weights)!=set(params):raise ValueError('adapter parameter set differs')
        with torch.no_grad():
            for name,p in params.items():p.copy_(weights[name])
        self.policy.critic.load_state_dict(torch.load(path/'critic.pt',map_location=self.policy.device_name,weights_only=True),strict=True)
        saved=torch.load(path/'optimizers.pt',map_location=self.policy.device_name,weights_only=True)
        for name,opt in self.optimizers.items():opt.load_state_dict(saved[name])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        if hasattr(self.policy,'execution_cache'):self.policy.execution_cache.revisions.clear()
        if hasattr(self.policy,'clear_prefix_cache'):self.policy.clear_prefix_cache()
        if self.state()!=meta['state'] or self.optimizer_identity()!=meta['optimizer']:
            raise ValueError('restored model/optimizer differs')
        self.last_update=meta['update']
        return meta


def _build_policy(config, device):
    """The only real model load boundary; called only by an explicit runtime."""
    import torch
    torch.cuda.set_device(device)
    # Select device BEFORE importing policy modules that initialize CUDA kernels.
    from agent.m9_policy import Policy, verify_checkpoint
    paths=config['model_paths']
    _read_json(Path(paths['warm_checkpoint'])/'manifest.json')
    manifest=verify_checkpoint(paths['warm_checkpoint'])
    if manifest['actor_sha256']!=EXPECTED_ACTOR:raise ValueError('wrong shared combined_0064 actor')
    warm=_read_json(Path(paths['warm_checkpoint'])/'policy.json')
    for key,expected in dict(contract='m9_research_v1',map_mode='infinite_freeze',navigation_steps=4,
        state_variant='reserves',reward_variant='bottleneck',gamma_tick=.99985,critic_features=108).items():
        if warm.get(key)!=expected:raise ValueError('warm checkpoint identity: '+key)
    c=SimpleNamespace(**{**DEFAULTS,**paths,'rng_seed':config['seed']})

    class OnlinePolicy(Policy):
        def values(self,prepared):return critic_values(self,prepared)
        def metadata(self):
            result=super().metadata()
            result.update(base_contract='m9_research_v1',contract=CONTRACT,profile='mea_online',
                critic_features=112,critic_feature_names=result['critic_feature_names']+['mea_'+k for k in FEATURE_NAMES],
                state2_source=config['source'],state2_association='one_fresh_transport_window_per_decision',
                coverage_basis='filtered_transport_frames',critic_execution_layout='per_row_fp32',
                critic_initialization='combined_0064_108_to112_new_columns_zero',
                alpha=1.,r2_enabled=False,stimulation=False,realtime_training=config['source']=='live')
            return result

    policy=OnlinePolicy(c,reference=True)
    if policy.digest()!=EXPECTED_ACTOR or policy.reference_digest()!=EXPECTED_ACTOR:
        raise ValueError('warm actor/reference identity differs')
    if any('lora_' not in n for n,p in policy.model.named_parameters() if p.requires_grad):
        raise ValueError('only LoRA parameters may train')
    # Check the historical actor under its original representation before adding
    # state2; a migration error must not be hidden by fresh rollout collection.
    probe=_read_json(Path(paths['warm_checkpoint'])/'reload_probe.json')
    prepared=Policy.prepare(policy,[r['observation'] for r in probe['rows']])
    logps=[]
    with torch.no_grad():
        for start in range(0,len(prepared),4):
            pp=prepared[start:start+4];rr=probe['rows'][start:start+4]
            v=torch.tensor([r['verb'] for r in rr],device=policy.device_name)
            t=torch.tensor([r['target'] for r in rr],device=policy.device_name)
            vd,td=policy.distributions(pp,v.tolist())
            logps.extend((vd.log_prob(v)+td.log_prob(t)).tolist())
    error=_difference(logps,probe['logp'])
    if error>1e-6:raise ValueError('historical warm actor migration drift: '+str(error))
    expand_online_critic(policy.critic)
    policy.migration_probe=dict(old_actor_logp_max_error=error,rows=len(prepared))
    policy.assert_reference_frozen()
    return policy


class M9Runtime:
    def __init__(self,config,output_dir):
        self.config=validate_runtime_config(config)
        self.output_dir=Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.closed=False;self.failed=False;self.env=None;self.replicas=None;self.core=None;self.tracker=None
        self.last_end=None;self.used_requests=set();self.used_snapshots=set()
        try:
            if len(config['devices'])>1:
                from .distributed import ReplicaGroup
                self.replicas=ReplicaGroup(config,self.output_dir)
            self.core=_ModelCore(_build_policy(config,config['devices'][0]),config,0,len(config['devices']))
            from agent.m9_env import GodotEnv
            self.env=GodotEnv(config['godot'],config['project'],map_mode=DEFAULTS['map_mode'],
                              navigation_steps=DEFAULTS['navigation_steps'])
        except BaseException:
            self.failed=True
            self.close()
            raise

    def _active(self):
        if self.closed or self.failed:raise RuntimeError('runtime is closed or failed')

    def reset(self,seed):
        self._active()
        from agent.m9_contract import Reward
        result=self.env.reset(seed)
        self.tracker=Reward(result['observation'],DEFAULTS['reward_variant'],DEFAULTS['gamma_tick'])
        return deepcopy(result['observation'])

    def prepare(self,observation,snapshot):
        self._active()
        row=project_observation(observation,snapshot,self.config)
        if (snapshot['request_id'] in self.used_requests or snapshot['snapshot_id'] in self.used_snapshots
            or self.last_end is not None and snapshot['start_frame']<self.last_end):
            raise ValueError('replayed or overlapping online snapshot')
        ids=self.core.policy._tokens(row['messages'])
        row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={})
        row['prepared_sha256']=digest(row)
        self.used_requests.add(snapshot['request_id']);self.used_snapshots.add(snapshot['snapshot_id'])
        self.last_end=snapshot['end_frame']
        return row

    def act(self,prepared):
        import torch
        self._active()
        with torch.no_grad():sample=self.core.policy.act([_materialize(self.core.policy,prepared)],with_reference=True)
        result={key:sample[key+'s' if key in ('action','verb','target','value') else key][0]
                for key in ('action','verb','target','value','logp','verb_logp','reference_verb','reference_target')}
        canonical(result)
        return result

    def value(self,prepared):
        import torch
        self._active()
        with torch.no_grad():value=float(self.core.policy.values([_materialize(self.core.policy,prepared)])[0])
        return _number(value,'bootstrap value')

    def step(self,action):
        self._active()
        if self.tracker is None:raise RuntimeError('reset required before step')
        nxt=self.env.step(action)
        score=self.tracker.score(nxt['transition'],nxt['observation'])
        return dict(observation=nxt['observation'],transition=nxt['transition'],
                    reward=score['reward'],duration=score['elapsed_ticks'],terminal=nxt['terminated'],
                    r1_score=score,r2_enabled=False,stimulation=False)

    def update(self,rows,bootstrap,update_id):
        self._active()
        try:
            if self.replicas:self.replicas.send(dict(op='update',rows=rows,bootstrap=bootstrap,update_id=update_id))
            return self.core.update(rows,bootstrap,update_id)
        except BaseException:
            self.failed=True
            if self.replicas:self.replicas.abort()
            raise

    def checkpoint(self,path,registration_sha,update_id):
        self._active()
        try:
            if self.replicas:
                self.replicas.send(dict(op='checkpoint',path=str(Path(path).resolve()),registration_sha=registration_sha,update_id=update_id))
            result=self.core.checkpoint(path,registration_sha,update_id)
            if self.replicas:
                self.replicas.checkpoint_barrier()
                self.core.collective.require(True,'worker checkpoint roundtrip')
                self.core.collective.identical(dict(model=self.core.state(),optimizer=self.core.optimizer_identity()),'checkpoint reload')
            return result
        except BaseException:
            self.failed=True
            if self.replicas:self.replicas.abort()
            raise

    def state(self):
        self._active()
        return self.core.state()

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.env is not None:self.env.close()
        finally:
            if self.replicas:
                if self.failed:self.replicas.abort()
                else:self.replicas.close()
