"""Final acceptance gates; absence of independently verified evidence is a failure."""
import copy,json,math,sys
from pathlib import Path
EXPECTED='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
def validate(report,independent):
    assert report['host']=='researcher-local-6'
    assert report['alpha']==1 and report['r2_enabled'] is False and report['state2_enabled'] is False
    assert report['initial_actor']==EXPECTED and report['final_actor']!=EXPECTED
    assert report['fresh_destination_rollout'] is True
    assert math.isfinite(report['cross_host_probe_error']) and report['cross_host_probe_error']>=0
    assert report['cross_host_probe_passed']==(report['cross_host_probe_error']<=1e-6)
    assert type(report['decisions']) is int and 1<=report['decisions']<=32
    assert report['decisions']==32 or report['terminal'] is True
    assert report['update_calls']==1 and report['metrics']['optimizer_steps']>=1
    required=['actor_changed','critic_changed','frozen_base_unchanged','reference_unchanged','inputs_unchanged','finite_parameters','no_frozen_gradients','sources_unchanged']
    assert set(report['checks'])==set(required) and all(report['checks'][k] is True for k in required)
    for key in ['initial_probe_error','pre_update_statistic_error','roundtrip_error']:
        assert math.isfinite(report[key]) and 0<=report[key]<=1e-6,key
    for key in ['actor_grad_norm','critic_grad_norm','policy_loss','critic_loss']:
        assert math.isfinite(report['metrics'][key]),key
    assert independent['passed'] is True and independent['decisions']==report['decisions']
    assert independent['final_actor']==report['final_actor'] and independent['initial_actor']==EXPECTED
    assert independent['reward_exact_matches']==report['decisions'] and independent['native_transitions']==report['decisions']
    assert independent['input_identity_verified'] and independent['gae_verified'] and independent['sources_unchanged']
    for key in ['old_statistic_error','new_statistic_error','bootstrap_error']:
        assert math.isfinite(independent[key]) and 0<=independent[key]<=1e-6,key
    assert independent['collection_pid']!=independent['verification_pid']
    return True
def self_test():
    r=dict(host='researcher-local-6',alpha=1,r2_enabled=False,state2_enabled=False,initial_actor=EXPECTED,final_actor='new',decisions=32,terminal=False,update_calls=1,
      metrics=dict(optimizer_steps=2,actor_grad_norm=1.,critic_grad_norm=1.,policy_loss=.1,critic_loss=.1),
      checks={k:True for k in ['actor_changed','critic_changed','frozen_base_unchanged','reference_unchanged','inputs_unchanged','finite_parameters','no_frozen_gradients','sources_unchanged']},
      initial_probe_error=0.,pre_update_statistic_error=0.,roundtrip_error=0.,fresh_destination_rollout=True,
      cross_host_probe_error=.07,cross_host_probe_passed=False)
    v=dict(passed=True,decisions=32,initial_actor=EXPECTED,final_actor='new',reward_exact_matches=32,native_transitions=32,
      input_identity_verified=True,gae_verified=True,sources_unchanged=True,old_statistic_error=0.,new_statistic_error=0.,bootstrap_error=0.,collection_pid=1,verification_pid=2)
    assert validate(r,v)
    changes=[('alpha',.5),('initial_actor','wrong'),('final_actor',EXPECTED),('decisions',0),('initial_probe_error',float('nan')),('roundtrip_error',1e-5)]
    for key,value in changes:
        bad=copy.deepcopy(r);bad[key]=value
        try:validate(bad,v)
        except AssertionError:pass
        else:raise AssertionError('accepted invalid '+key)
    for name in r['checks']:
        bad=copy.deepcopy(r);bad['checks'][name]=False
        try:validate(bad,v)
        except AssertionError:pass
        else:raise AssertionError('accepted missing invariant '+name)
    bad=copy.deepcopy(v);bad['new_statistic_error']=float('nan')
    try:validate(r,bad)
    except AssertionError:pass
    else:raise AssertionError('accepted nonfinite verification')
    print('acceptance negative checks: 15 rejected; valid fixture accepted (not GPU evidence)')
if __name__=='__main__':
    if sys.argv[1]=='--self-test':self_test()
    else:
        root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text());v=json.loads((root/'independent_reload.json').read_text())
        validate(r,v);print(json.dumps(dict(passed=True,decisions=r['decisions'],optimizer_steps=r['metrics']['optimizer_steps'],initial_actor=r['initial_actor'],final_actor=r['final_actor'])))
