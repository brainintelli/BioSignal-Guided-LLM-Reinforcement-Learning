"""Offline representation candidate. No inference, training or engine execution.

Reads facts already exposed to the old actor. Raw legal actions/landmark IDs only
bind semantic labels to engine actions. This is NOT a replacement PPO policy.
"""
from collections import Counter
from copy import deepcopy
import json

VERBS = dict(move='移动', navigate='前往', gather='采集', eat='吃饭',
             drink='喝水', rest='休息', build_camp='建造', wait='等待')
KINDS = dict(tree='树木', pine='松树', rock='岩石', berry='浆果',
             mushroom='蘑菇', water='饮水岸边', camp='营地', threat='威胁')
TERRAIN = {'?':'未观察', '~':'水面不可走', '#':'实体阻挡', '.':'草地',
           's':'沙地', 'f':'林地', '*':'花地'}
DIRECTIONS = {(0,-1):'向上', (1,0):'向右', (0,1):'向下', (-1,0):'向左'}
RULES = (
    '选择一个合法中文动作，只输出动作名称。需要目标时再从该动作的合法目标中选择。'
    '每种动作是一个选择，目标数量不是优先级。移动的方向固定为地图上、右、下、左，'
    '每次只走相邻一格。前往是朝选定地标避障走一格，不自动喝水或采集。'
    '建造表示建营地。格是距离单位，刻是世界时间；动作跨多刻时逐刻结算。'
    '身体数值是剩余量，0空100满。每刻饱食减0.20、水分减0.08，移动耗0.35体力。'
    '一天180刻，前140刻非夜间、后40刻夜间；先推进时间再结算消耗和伤害。'
    '营地外夜间每刻伤0.10；饱食为0伤1.2、水分为0伤1.8、体力为0伤0.6。'
    '营地外敌人距离小于1.6格时每刻近身伤1.2。'
    '采集树木或松树得2木材，岩石得2石头，浆果或蘑菇得3食物，耗2刻和2体力，90刻再生。'
    '吃饭用1食物补18饱食，喝水在岸边补50水分，各耗1刻。休息补14体力耗3刻；'
    '营地2.5格内且开始时饱食、水分均至少20，另补2生命。'
    '建造耗4木材2石头和4刻；移动、前往、等待各耗1刻，满值操作也耗时。'
    '所有位置相对现在的角色，地图上方是北；地形为以脚下为中心左右上下各2格的25格。'
    '未观察不等于可走，水面和实体不可走；饮水岸边是可站立地点，不是水面。'
    '已知路程是到目标或可互动位置的路径长度，-1表示未提供或不可达。'
    '最近动作的位置指动作完成位置相对现在的位置。列表次序不是建议，没有推荐目标。'
)
CONCISE_RULES = (
    '从合法动作选一个中文名称，只输出该名称；需要目标时再选合法目标。目标数量不是优先级。'
    '移动：向上/向右/向下/向左走相邻一格，方向固定为地图方向。前往：朝选定地标避障走一格，不自动互动。'
    '建造指建营地。格是距离，刻是时间。身体数值0空100满。'
    '每刻饱食减.20、水分减.08。移动耗.35体力。一天180刻，140起夜间，先推进时间再结算。'
    '营地外夜间每刻伤.10，饱食0伤1.2、水分0伤1.8、体力0伤.6；营地外敌距小于1.6格每刻伤1.2。'
    '采集树木/松树得2木材，岩石得2石头，浆果/蘑菇得3食物，耗2刻2体力，90刻再生。'
    '吃饭耗1食物补18饱食，喝水岸边补50水分，各1刻。休息补14体力耗3刻；'
    '营地2.5格内且开始时饱食、水分都至少20，另补2生命。'
    '建造耗4木2石和4刻；移动/前往/等待各1刻，满值操作仍耗时。'
    '位置均相对现在的角色，上方是北；周围地形覆盖上下左右各2格共25格。'
    '未观察不等于可走，水面和实体不可走，饮水岸边可站立。'
    '路程为已知路径至目标或互动位置的格数，-1为未提供或不可达。'
    '就绪不代表能采集；可选项以合法动作及地标标注为准。最近动作的落点相对现在的位置。次序不是建议。'
)


def position(dx, dy):
    parts = []
    if dx:
        parts.append(('右' if dx > 0 else '左') + str(abs(dx)) + '格')
    if dy:
        parts.append(('下' if dy > 0 else '上') + str(abs(dy)) + '格')
    return '、'.join(parts) or '脚下'


POSITIONS = {position(x, y):(x, y) for y in range(-2,3) for x in range(-2,3)}


def time_state(tick):
    t = int(tick) % 180
    phase = '清晨' if t < 20 else '白天' if t < 115 else '黄昏' if t < 140 else '夜间'
    return {'时段':phase, **({'距离入夜还剩刻数':140-t} if t < 140 else {'距离天亮还剩刻数':180-t})}


def time_tick(value):
    return 140-value['距离入夜还剩刻数'] if '距离入夜还剩刻数' in value else 180-value['距离天亮还剩刻数']


def map_state(rows, mode='sparse'):
    if len(rows) != 5 or any(len(row) != 5 for row in rows):
        raise ValueError('Expected complete 5 by 5 map')
    if mode == 'grid':
        return {'各行': [[TERRAIN[c] for c in row] for row in rows]}
    if mode != 'sparse':
        raise ValueError(mode)
    counts = Counter(''.join(rows))
    default = sorted(counts, key=lambda c:(-counts[c], c))[0]
    return {'未列位置均为':TERRAIN[default], '例外位置':{
        position(x-2,y-2):TERRAIN[c] for y,row in enumerate(rows)
        for x,c in enumerate(row) if c != default}}


def map_rows(value):
    reverse = {v:k for k,v in TERRAIN.items()}
    if '各行' in value:
        return [''.join(reverse[c] for c in row) for row in value['各行']]
    rows = [[reverse[value['未列位置均为']]]*5 for _ in range(5)]
    for location, kind in value['例外位置'].items():
        x, y = POSITIONS[location]
        rows[y+2][x+2] = reverse[kind]
    return [''.join(row) for row in rows]


def visit_state(rows):
    return {position(x-2,y-2):n for y,row in enumerate(rows) for x,n in enumerate(row) if n}


def visit_rows(value):
    rows = [[0]*5 for _ in range(5)]
    for location, count in value.items():
        x,y = POSITIONS[location]
        rows[y+2][x+2] = count
    return rows


def build(record, mode='sparse'):
    old = json.loads(record['model_input'][1]['content'])
    obs = record['observation']
    player = obs['player']
    # Coordinates/kinds in actor facts provide semantic names. Raw IDs are kept
    # exclusively in the execution catalog, outside the actor prompt.
    alias_names, landmarks = {}, []
    for alias, kind, dx, dy, path, available, ready in old['地标[别名,种类,dx,dy,路程,可用,再生步]']:
        name = KINDS.get(kind, kind) + '（' + position(dx,dy) + '）'
        alias_names[alias] = name
        affordances = []
        for code, semantic in [('go','前往'),('take','采集')]:
            if any(row[1] == alias for row in old['各动作的合法目标'].get(code, [])):
                affordances.append(semantic)
        landmarks.append({'名称':name, '已知路程':path, '可用':bool(available),
                          '再生还需刻数':ready, '可执行':affordances})
    if len(set(alias_names.values())) != len(alias_names):
        raise ValueError('Semantic target collision; never silently merge actions')
    legal_directions = [DIRECTIONS[tuple(row[1:])] for row in old['各动作的合法目标'].get('move',[])]
    code_to_name = dict(zip(('move','go','take','eat','drink','rest','camp','wait'), VERBS.values()))
    history = []
    for verb,dx,dy,target,delta in old['过去[动作,dx,dy,目标种类相对位置,变化]']:
        history.append({'动作':VERBS.get(verb,verb), '完成位置':position(dx,dy),
                        '目标':None if target is None else KINDS.get(target[0],target[0])+'（'+position(*target[1:])+'）',
                        '变化':{dict(health='生命',hunger='饱食',thirst='水分',energy='体力').get(k,k):v for k,v in delta.items()}})
    state = {'身体':deepcopy(old['剩余数值']), '背包':deepcopy(old['背包']),
             '时间':time_state(old['日内步']),
             '营地':None if old['营地相对位置'] is None else position(*old['营地相对位置']),
             '威胁':[{'种类':KINDS.get(t[0],t[0]),'位置':position(t[1],t[2]),'距离':t[3]} for t in old['危险']],
             '地形':map_state(old['局部5x5'],mode),
             '走过的位置':visit_state(old['局部踏入次数5x5']), '地标':landmarks,
             '最近动作':history, '合法动作':[code_to_name[v] for v in old['合法动作代码']],
             '合法移动方向':legal_directions}
    ids = sorted({str(t['id']) for t in obs.get('nearby', [])} |
                 {str(a['target_id']) for a in obs['allowed_actions'] if 'target_id' in a and a['verb']!='move'})
    id_names = {key:alias_names.get('t'+str(i)) for i,key in enumerate(ids)}
    catalog = []
    for action in obs['allowed_actions']:
        verb = action['verb']
        target = None
        if verb == 'move':
            _,x,y = action['target_id'].split('_')
            delta = int(x)-int(player['x']), int(y)-int(player['y'])
            target = DIRECTIONS[delta]
            text = target+'走1格'
        elif verb in ('navigate','gather'):
            target = id_names[action['target_id']]
            if target is None:
                raise ValueError('No exposed semantic descriptor for legal target')
            text = VERBS[verb]+target+('，走1格' if verb=='navigate' else '')
        else:
            text = {'eat':'吃1份食物', 'drink':'原地喝水', 'rest':'原地休息3刻',
                    'build_camp':'建造营地', 'wait':'原地等待1刻'}[verb]
        catalog.append({'verb':VERBS[verb], 'target':target, 'text':text, 'engine_action':deepcopy(action)})
    if len({a['text'] for a in catalog}) != len(catalog):
        raise ValueError('Semantic action collision')
    return {'state':state, 'messages':[{'role':'system','content':RULES},
                                     {'role':'user','content':render(state)}],
            'action_catalog':catalog, 'metadata':{'source_profile':old['环境'],
             'candidate':'semantic_state_v1_offline', 'map_mode':mode, 'not_a_training_policy':True}}


def render(state):
    lines = ['身体（剩余量，满值100）：'+ '，'.join(f'{k}{v:g}' for k,v in state['身体'].items())+'。',
             '背包：'+'，'.join(f'{k}{v}' for k,v in state['背包'].items())+'。']
    time = state['时间']
    remaining = '距离入夜还剩刻数' if '距离入夜还剩刻数' in time else '距离天亮还剩刻数'
    lines.append('时间：'+time['时段']+'，'+remaining.replace('刻数','')+str(time[remaining])+'刻。')
    lines.append('营地：'+(state['营地'] or '尚未建造')+'。')
    lines.append('可见威胁：'+('；'.join(f"{x['种类']}在{x['位置']}，距离{x['距离']}格" for x in state['威胁']) or '无')+'。')
    lines.append('合法移动方向：'+'、'.join(state['合法移动方向'])+'；每次走1格。')
    terrain = state['地形']
    if '各行' in terrain:
        lines.append('周围地形：每行从左2格、左1格、脚下所在列、右1格到右2格。')
        for name,row in zip(('上2格这一行','上1格这一行','脚下这一行','下1格这一行','下2格这一行'),terrain['各行']):
            lines.append(name+'：'+'、'.join(row)+'。')
    else:
        lines.append('周围地形：25格中未列位置均为'+terrain['未列位置均为']+'；例外如下。')
        lines += [location+'：'+kind+'。' for location,kind in terrain['例外位置'].items()]
    lines.append('踏入次数（未列位置为0）：'+('；'.join(f'{k}{v}次' for k,v in state['走过的位置'].items()) or '无')+'。')
    lines.append('地标（已知路程以格计；可用表示资源就绪，不代表已能采集）：')
    for obj in state['地标']:
        lines.append(obj['名称']+'：已知路程'+str(obj['已知路程'])+'；'+('可用' if obj['可用'] else '尚未就绪')+
                     '；再生还需'+str(obj['再生还需刻数'])+'刻；可执行'+('、'.join(obj['可执行']) or '无')+'。')
    lines.append('最近动作（按发生顺序）：')
    if not state['最近动作']:
        lines.append('无。')
    for event in state['最近动作']:
        changes = '，'.join(f'{k}{v:+g}' for k,v in event['变化'].items()) or '无数值变化'
        lines.append(event['动作']+'；完成位置'+event['完成位置']+'；目标'+(event['目标'] or '未提供')+'；'+changes+'。')
    lines.append('现在可选动作：'+'、'.join(state['合法动作'])+'。')
    return '\n'.join(lines)


def render_concise(state):
    """Same structured facts; remove repeated wording and explicit zero cooldowns."""
    lines = ['身体：'+'，'.join(f'{k}{v:g}' for k,v in state['身体'].items())+'。',
             '背包：'+'，'.join(f'{k}{v}' for k,v in state['背包'].items())+'。']
    t = state['时间']
    if '距离入夜还剩刻数' in t:
        clock = '距入夜'+str(t['距离入夜还剩刻数'])+'刻'
    else:
        clock = '距天亮'+str(t['距离天亮还剩刻数'])+'刻'
    lines += ['时间：'+t['时段']+'，'+clock+'。', '营地：'+(state['营地'] or '未建造')+'。',
              '可见威胁：'+('；'.join(f"{x['种类']}（{x['位置']}），距离{x['距离']}格" for x in state['威胁']) or '无')+'。',
              '能移动：'+'、'.join(state['合法移动方向'])+'。']
    terrain = state['地形']
    if '各行' in terrain:
        raise ValueError('Concise variant uses explicit default plus exceptions')
    lines.append('周围地形：未列位置都是'+terrain['未列位置均为']+'；'+
                 '；'.join(k+'为'+v for k,v in terrain['例外位置'].items())+'。')
    lines.append('踏入次数：'+('，'.join(f'{k}{v}次' for k,v in state['走过的位置'].items()) or '无')+'；未列位置为0。')
    lines.append('地标（未写再生表示0刻）：')
    for obj in state['地标']:
        ready = ('；再生'+str(obj['再生还需刻数'])+'刻') if obj['再生还需刻数'] else ''
        lines.append(obj['名称']+'：路程'+str(obj['已知路程'])+'，'+('就绪' if obj['可用'] else '未就绪')+ready+
                     '，'+('可'+'、'.join(obj['可执行']) if obj['可执行'] else '不能前往或采集')+'。')
    lines.append('最近动作：')
    if not state['最近动作']:
        lines.append('无。')
    for event in state['最近动作']:
        target = event['目标'] or '（目标未提供）'
        changes = '、'.join(f'{k}{v:+g}' for k,v in event['变化'].items()) or '数值未变'
        lines.append(event['动作']+target+'，落点'+event['完成位置']+'，'+changes+'。')
    lines.append('合法动作：'+'、'.join(state['合法动作'])+'。')
    return '\n'.join(lines)
