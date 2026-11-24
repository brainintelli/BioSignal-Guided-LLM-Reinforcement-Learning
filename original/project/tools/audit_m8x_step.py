"""Independent reconstruction of an open3 step, not a release authorization.

The caller supplies a freshly replayed engine result, not the recorded one.
Whole-run source, episode, RNG and budget checks are separate requirements.
"""
from agent import m8x_contract as contract
from agent.rl_contract import transition_at_horizon
from tools.audit_m6_run import assert_equivalent


def validate_step(row, observation, actual_next, tracker, *, assessment=False, horizon=1800):
    assert_equivalent(observation,row['observation'],'observation')
    assert_equivalent(actual_next['observation'],row['next_observation'],'next_observation')
    assert_equivalent(actual_next['transition'],row['transition'],'transition')
    if row.get('salt')!='m6' or row.get('contract')!=contract.VERSION or row.get('source')!='model':
        raise ValueError('Changed actor contract, salt or action provenance')
    encoded=contract.encode(observation)
    assert_equivalent(encoded['messages'],row['model_input'],'model_input')
    v,t=row['verb_index'],row['target_index']
    if type(v) is not int or not 0<=v<len(encoded['branches']):
        raise ValueError('Invalid verb index')
    branch=encoded['branches'][v]
    if type(t) is not int or not 0<=t<len(branch):
        raise ValueError('Invalid target index')
    assert_equivalent(branch[t],row['action'],'action')
    target_input=contract.target_messages(encoded,v) if len(branch)>1 else None
    assert_equivalent(target_input,row['target_input'],'target_input')
    actual_transition=actual_next['transition']
    terminal=actual_transition['after']['dead']
    if type(actual_next['terminated']) is not bool or actual_next['terminated']!=terminal:
        raise ValueError('Engine terminal flag disagrees with the real final snapshot')
    reward=tracker.score(actual_transition,actual_next['observation'])
    assert_equivalent(reward,row['reward'],'reward')
    if assessment:
        if (row.get('reward_scope')!='full_executed_transition'
                or row.get('scored_horizon')!=horizon
                or type(row.get('engine_terminated')) is not bool
                or row['engine_terminated']!=terminal):
            raise ValueError('Assessment scoring scope differs from executed transition')
        scored=transition_at_horizon(actual_transition,horizon)
    else:
        scored=actual_transition
    scored_terminal=scored['after']['dead']
    truncated=assessment and scored['after']['tick']>=horizon and not scored_terminal
    if type(row['terminal']) is not bool or type(row['truncated']) is not bool:
        raise ValueError('Terminal and truncation flags must be boolean')
    assert_equivalent(scored_terminal,row['terminal'],'terminal')
    assert_equivalent(truncated,row['truncated'],'truncated')
    return scored_terminal or truncated,scored,reward
