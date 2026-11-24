"""Semantic, information-equivalent actor input; original engine actions retained."""
from agent import m8x_contract as old
from agent.m8s_state import build,render_concise,CONCISE_RULES

VERSION='open_world_semantic_v1'
CODES=('移动','前往','采集','吃饭','喝水','休息','建造','等待')
VERBS=old.VERBS
REWARD_CONFIG=old.REWARD_CONFIG
SurvivalReward=old.SurvivalReward


def encode(observation,salt='m6'):
    previous=old.encode(observation,salt)
    candidate=build({'observation':observation,'model_input':previous['messages']})
    by_action={old.legacy.canonical(x['engine_action']):x for x in candidate['action_catalog']}
    targets=[]
    for branch in previous['branches']:
        targets.append([by_action[old.legacy.canonical(a)]['target'] or '' for a in branch])
    return dict(messages=[{'role':'system','content':CONCISE_RULES},
                          {'role':'user','content':render_concise(candidate['state'])}],
                branches=previous['branches'],verb_mask=previous['verb_mask'],
                features=old.critic_features(previous),target_texts=targets,
                semantic_state=candidate['state'],contract=VERSION,salt=salt)


def target_messages(row,verb):
    if not row['branches'][verb]:raise ValueError('Illegal branch')
    instruction=('从上面能移动的方向中选一个，只输出向上、向右、向下或向左。' if verb==0 else
                 '从上面标记可'+CODES[verb]+'的地标中选择一个，只输出完整地标名称（包含位置）。')
    return row['messages']+[{'role':'assistant','content':CODES[verb]},
                            {'role':'user','content':instruction}]
