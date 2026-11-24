"""Freeze real observation questions, plus actual Godot dawn/night snapshots."""
import argparse,hashlib,json,random,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8x_contract import encode
from agent.m8s_state import KINDS,position

COMMON_RULES=(
 '你在做游戏环境事实判断，不是在选择生存策略。只回答是或否。'
 '所有相对位置以当前角色为原点，x向右、y向下；上下左右移动一次仅一格。'
 '一天180刻，日内步140至179为夜间。地形范围是上下左右各2格，共25格。'
 '旧地图每行从左到右、各行从上到下，中心为角色；~水面、#实体、?未观察、.草地、s沙地、f林地、*花地。'
 '旧地标数组列为别名、种类、dx、dy、路程、可用、再生步。tree树木、pine松树、rock岩石、berry浆果、mushroom蘑菇、water饮水岸边。'
 '旧动作move移动、go导航、take采集、eat吃饭、drink喝水、rest休息、camp建造、wait等待。'
 '能否执行以合法动作及目标为准；资源可用不等于能够采集。'
 '历史位置是过去动作完成的位置相对现在的位置，最后一项是最近一次动作。'
 '未列出的地形以明确写出的默认地形为准，未列踏入次数为0。')


def candidates(observation):
    facts=encode(observation)['facts']
    actions=observation['allowed_actions'];p=observation['player'];out=[]
    def add(category,text,answer):out.append(dict(category=category,question=text,answer=bool(answer)))
    for dx,dy,direction in ((0,-1,'上'),(1,0,'右'),(0,1,'下'),(-1,0,'左')):
        action={'verb':'move','target_id':f"tile_{p['x']+dx}_{p['y']+dy}"}
        add('move',f'现在是否可以向{direction}移动一格？',action in actions)
    for y in range(5):
        for x in range(5):
            add('terrain',f'{position(x-2,y-2)}的地形是否为水面？',facts['局部5x5'][y][x]=='~')
    for kind in ('tree','pine','rock','berry','mushroom'):
        for side in ('左','右'):
            add('resource_side',f'当前观察中是否有{KINDS[kind]}位于角色的{side}侧？',
                any(r[1]==kind and (r[2]<0 if side=='左' else r[2]>0) for r in facts['地标[别名,种类,dx,dy,路程,可用,再生步]']))
    add('drink','现在是否可以原地喝水？',{'verb':'drink'} in actions)
    add('time','再经过3刻后是否处于夜间？',(int(observation['tick'])+3)%180>=140)
    add('inventory','现在背包中的食物是否至少有2份？',facts['背包']['食物']>=2)
    for target in observation['nearby']:
        if target['kind'] in ('water','camp'):continue
        name=KINDS[target['kind']]+'（'+position(target['x']-p['x'],target['y']-p['y'])+'）'
        add('gather',f'现在是否可以采集{name}？',{'verb':'gather','target_id':target['id']} in actions)
    history=facts['过去[动作,dx,dy,目标种类相对位置,变化]']
    add('history','最近一次已完成的动作是否为移动或前往？',bool(history) and history[-1][0] in ('move','navigate'))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--godot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();project=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    from agent.m8x_env import GodotEnv
    records=[json.loads(x) for x in (project/'output/rl_kv2_20261107/optimized_transitions.jsonl').read_text().splitlines()]
    observations=[{'id':f"recorded_{i}",'source':'recorded_model_rollout','seed':r['seed'],'observation':r['observation']} for i,r in enumerate(records)]
    snapshots=[]
    for seed in (884910,884911,884912,884913):
        with GodotEnv(a.godot,a.project) as env:
            state=env.reset(seed,'open3')
            for t in range(181):
                if t in (0,20,114,115,136,137,138,139,140,177,178,179,180):
                    entry={'id':f'wait_{seed}_{t}','source':'scripted_wait_real_Godot','seed':seed,'observation':state['observation']}
                    snapshots.append(entry);observations.append(entry)
                if t<180:
                    state=env.step({'verb':'wait'})
                    if state['terminated']:raise RuntimeError('Snapshot script ended before planned time boundary')
    questions=[]
    for row in observations:
        for q in candidates(row['observation']):
            identity=json.dumps([row['id'],q],ensure_ascii=False,sort_keys=True)
            questions.append({**q,'observation_id':row['id'],'id':hashlib.sha256(identity.encode()).hexdigest()})
    selected=[]
    for category in sorted({q['category'] for q in questions}):
        for answer in (False,True):
            pool=sorted((q for q in questions if q['category']==category and q['answer']==answer),key=lambda q:q['id'])
            if len(pool)<6:raise ValueError(f'Insufficient balanced questions: {category} {answer}')
            selected.extend(pool[:6])
    grounding=[]
    for direction,delta in zip(('向上','向右','向下','向左'),((0,-1),(1,0),(0,1),(-1,0))):
        pool=[]
        for row in observations[:384]:
            p=row['observation']['player'];dx,dy=delta
            action={'verb':'move','target_id':f"tile_{p['x']+dx}_{p['y']+dy}"}
            if action in row['observation']['allowed_actions']:
                key=hashlib.sha256((direction+row['id']).encode()).hexdigest()
                pool.append(dict(id=key,observation_id=row['id'],direction=direction,expected_action=action))
        grounding.extend(sorted(pool,key=lambda x:x['id'])[:8])
    result=dict(observations=observations,questions=selected,grounding=grounding,common_rules=COMMON_RULES,
                provenance={'recorded_model_states':384,'new_real_scripted_snapshots':len(snapshots),
                            'scripted_policy':'wait only; no model training','questions_per_category':12,
                            'answers_balanced_per_category':True,'seed_selection':'sha256 order, fixed before model scoring'})
    (out/'dataset.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'observations':len(observations),'questions':len(selected),'grounding':len(grounding),'categories':dict(Counter(q['category'] for q in selected))}))

if __name__=='__main__':main()
