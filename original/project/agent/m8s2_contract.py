"""Less repeated Chinese prose, with the same observable facts and legal actions."""
from collections import defaultdict
from agent.m8s_contract import encode as previous_encode,CODES,VERBS,REWARD_CONFIG,SurvivalReward

VERSION='open_world_semantic_v2'
RULES=(
    '选择合法动作及其目标。移动目标为向上、向下、向左、向右，每次一格；'
    '前往地标也只避障走一格，不自动互动。位置相对自身，上为北，单位格。周围为上下左右各2格；'
    '未观察不等于可走，水面和实体不可走，饮水岸边可站立。身体0空100满。'
    '每刻推进后结算：饱食减0.2、水分减0.08；饱食、水分、体力为0分别伤1.2、1.8、0.6。'
    '每天180刻，140起夜间；营地外夜间每刻伤0.1，敌距小于1.6格每刻伤1.2。移动耗0.35体力。'
    '采集耗2刻2体力：树木或松树得2木材，岩石得2石头，浆果或蘑菇得3食物，90刻再生。'
    '吃饭用1食物补18饱食，岸边喝水补50水分，各1刻。休息3刻补14体力；营地2.5格内且行动前饱食水分都至少20，另补2生命。'
    '建造营地用4木材2石头耗4刻；移动、前往、等待各1刻，满值操作仍耗时。'
    '路程指到目标或互动处的已知路径，-1为未知或不可达。地标默认就绪、再生0刻；就绪不代表可采集。'
    '踏入次数未列为0。历史落点相对现在，未列变化为0。地标按标注选；次序和目标数无优先级。'
)

def terrain_text(terrain):
    groups=defaultdict(list)
    for location,kind in terrain['例外位置'].items():groups[kind].append(location)
    return '周围地形：其余'+terrain['未列位置均为']+'。'+'。'.join(kind+'：'+'；'.join(locations) for kind,locations in groups.items())

def render(state):
    t=state['时间'];night='距离入夜还剩刻数' in t
    lines=['身体：'+'，'.join(f'{k}{v:g}' for k,v in state['身体'].items()),
           '背包：'+'，'.join(f'{k}{v}' for k,v in state['背包'].items()),
           t['时段']+'，距'+('入夜' if night else '天亮')+str(t['距离入夜还剩刻数' if night else '距离天亮还剩刻数'])+'刻',
           '营地：'+(state['营地'] or '无'),
           '威胁：'+('；'.join(f"{x['种类']}（{x['位置']}），距离{x['距离']}" for x in state['威胁']) or '无'),
           '能移动：'+'、'.join(state['合法移动方向']),terrain_text(state['地形']),
           '踏入次数：'+('；'.join(f'{k}{v}次' for k,v in state['走过的位置'].items()) or '无'),'地标：']
    for obj in state['地标']:
        parts=['路程'+str(obj['已知路程']),('可'+'、'.join(obj['可执行'])) if obj['可执行'] else '不可前往或采集']
        if not obj['可用']:parts.append('未就绪')
        if obj['再生还需刻数']:parts.append('再生'+str(obj['再生还需刻数'])+'刻')
        lines.append(obj['名称']+'：'+'，'.join(parts))
    lines.append('最近动作：')
    for event in state['最近动作']:
        changes='，'.join(f'{k}{v:+g}' for k,v in event['变化'].items() if v)
        lines.append(event['动作']+(event['目标'] or '')+'，落点'+event['完成位置']+('，'+changes if changes else ''))
    if not state['最近动作']:lines.append('无')
    lines.append('合法动作：'+'、'.join(state['合法动作']))
    return '\n'.join(lines)

def encode(observation,salt='m6'):
    row=previous_encode(observation,salt)
    row['messages']=[{'role':'system','content':RULES},{'role':'user','content':render(row['semantic_state'])}]
    row['contract']=VERSION
    return row
