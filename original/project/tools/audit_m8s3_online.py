"""Independent selected-actor forward/RNG and actual rendered-game replay audit.

Run after the bounded server exits. Imports CUDA only after offline acceptance
and idle-GPU checks. The rendered32-step prefix is not a survival test.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.m8s3_online_gate import verify_offline_gate
from agent.m8s3_training import sha,verify_files
from agent.m8s3_server import decision_identity,validate_record
from agent.m6_live_session import SessionRegistry
from agent.m8x_server import verify_saved_cuda_draws
from tools.audit_m8s3_replay import Comparator
from tools.m8x_live_sampler_probe import require_idle_gpu,write_new
from tools.audit_m8x_live_journey import snapshot


def verify_client_steps(clients,servers,env,context):
    if not clients or len(clients)!=len(servers):raise ValueError('Client/server decision count differs')
    cmp=Comparator(1e-6)
    obs=env.reset(context['world_seed'],'open4')['observation']
    if obs['tick']!=0:raise ValueError('Actual game journey must begin at zero')
    for client,server in zip(clients,servers):
        d=server['decision']
        if (client.get('source')!='model' or client.get('profile')!='open4' or client.get('world_version')!=6
                or d['agent_context']!=context or server['agent_context']!=context):
            raise ValueError('Manual/fallback/other-world action in game log')
        cmp.same(obs,server['observation'],'full_game_observation')
        cmp.same(client['tick_before'],obs['tick'],'tick_before')
        cmp.same(client['action'],d['action'],'action')
        for key in ('actor_sha256','registration_sha256','agent_context'):
            cmp.same(client['model_decision'][key],d[key],'client_identity.'+key)
        for key,value in client['model_decision'].items():cmp.same(value,d[key],'client_response.'+key)
        nxt=env.step(d['action']);obs=nxt['observation'];after=nxt['transition']['after']
        cmp.same(client['tick_after'],after['tick'],'tick_after')
        cmp.same(client['vitals'],{k:after[k] for k in ('health','hunger','thirst')},'client_vitals')
        status=client['runtime_status']
        if (status.get('runtime_day_limit','missing') is not None or status.get('runtime_tick_limit','missing') is not None
                or status['state']!='running' or status['stop_reason']!='' or status['model_error']!='' or after['dead']):
            raise ValueError('Rendered prefix contains a stop/error or runtime horizon')
    return dict(passed=True,executed_decisions=len(clients),verified_ticks=obs['tick'],
                final_vitals={k:obs['player'][k] for k in ('health','hunger','thirst')},max_float_error=cmp.max_float_error)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('run','server','client','receipt','image','client-main','capture-driver','output'):
        ap.add_argument('--'+key,required=True)
    args=ap.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    gate=verify_offline_gate(args.run)
    verify_files(ROOT,json.loads((Path(args.run)/'registration.json').read_text()))
    receipt=json.loads(Path(args.receipt).read_text())
    identity=decision_identity(gate['actor_sha256'],gate['registration_sha256'],gate['config']['verb_exploration_epsilon'])
    for field in ('actor_sha256','registration_sha256','profile','world_version','environment_version','balance_sha256'):
        if receipt[field]!=identity[field]:raise ValueError('Rendered receipt identity differs')
    for key,path in [('client_sha256',args.client),('image_sha256',args.image),('main_sha256',args.client_main),('driver_sha256',args.capture_driver)]:
        if receipt[key]!=sha(path):raise ValueError('Rendered evidence changed: '+key)
    if (receipt['passed'] is not True or receipt['synthetic_model'] is not False or receipt['failures']!=0
            or receipt['decisions']!=32 or receipt['model_errors']!=0 or receipt['fallbacks']!=0
            or receipt['unexpected_collisions']!=0 or receipt['display_server']=='headless'
            or receipt['windowed'] is not True or receipt['width']<1000 or receipt['height']<600):
        raise ValueError('Actual rendered game acceptance failed')
    inputs={key:sha(getattr(args,key)) for key in ('server','client','receipt','image','client_main','capture_driver')}
    servers=snapshot(args.server,inputs['server']);clients=snapshot(args.client,inputs['client'])
    if not 32<=len(servers)<=128:raise ValueError('Unexpected bounded validation log size')
    game=[r for r in servers if r['agent_context']==receipt['agent_context']]
    from agent.m8s3_env import GodotEnv
    with GodotEnv(gate['config']['godot'],gate['config']['project']) as env:
        journey=verify_client_steps(clients,game,env,receipt['agent_context'])
    if journey['verified_ticks']!=receipt['ticks']:raise ValueError('Rendered final tick differs')
    Comparator(1e-6).same(journey['final_vitals'],receipt['vitals'],'rendered_final_vitals')
    require_idle_gpu()
    from tools.benchmark_m8s3 import setup
    setup()
    import torch
    from agent.m8s3_run import make_policy,load_checkpoint
    p=make_policy(SimpleNamespace(**gate['config']),reference=False)
    load_checkpoint(p,gate['checkpoint']);p.eval()
    if p.digest()!=gate['actor_sha256']:raise ValueError('Selected actor differs')
    class Stream:
        def __init__(self,seed):self.generator=torch.Generator(device='cuda').manual_seed(seed)
        def snapshot(self):return self.generator.get_state().cpu().numpy().tobytes()
        def restore(self,state):self.generator.set_state(torch.tensor(list(state),dtype=torch.uint8))
    sessions=SessionRegistry(Stream,lambda _:None)
    error=0.
    with torch.no_grad():
        for row in servers:
            validate_record(row,identity)
            d=row['decision'];prepared=p.prepare([row['observation']])
            with p._scope(prepared,False):
                vd=p.verb_distribution(prepared);td=p.target_distribution(prepared,[d['verb_index']])
            for actual,recorded in ((vd.probs[0].tolist(),d['verb_probabilities']),(td.probs[0].tolist(),d['target_probabilities'])):
                if len(actual)!=len(recorded):raise ValueError('Probability layout differs')
                error=max(error,max(abs(x-y) for x,y in zip(actual,recorded)))
            if error>1e-6:raise ValueError('Independent online forward probabilities differ')
            verify_saved_cuda_draws(row,sessions._entry(row['agent_context'])['generator'],torch)
            sessions.restore_records([row])
    if p.digest()!=gate['actor_sha256']:raise ValueError('Read-only audit changed actor')
    if inputs!={key:sha(getattr(args,key)) for key in inputs}:raise ValueError('Online evidence changed during audit')
    write_new(args.output,dict(passed=True,identity=identity,actual_cuda=True,model_forward_max_error=error,
        audited_online_decisions=len(servers),rng_draws_reproduced=2*len(servers),journey=journey,
        input_sha256=inputs,source_sha256=sha(__file__),weights_unchanged=True,milestone_achieved=False,
        scope='Offline acceptance plus independently rebuilt HTTP decisions/RNG and rendered32-step game prefix; GPU release still requires process-exit check'))
    print(json.dumps(dict(passed=True,online_decisions=len(servers),game_decisions=len(game),max_probability_error=error)),flush=True)


if __name__=='__main__':main()
