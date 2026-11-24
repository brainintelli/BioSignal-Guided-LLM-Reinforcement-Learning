"""Supply-margin reward and extra critic facts; the actor encoder is unchanged.

Food shaping is a bounded telescoping potential. The small per-tick reserve
cost deliberately adds a safety-margin preference; only the potential part is
policy invariant. Neither term chooses actions or reads unseen map cells.
"""
from copy import deepcopy
import math

from agent.m8s3_contract import (SurvivalReward, REWARD_CONFIG as ORIGINAL_CONFIG,
                                _number, _position_xy, validate_observation_identity)

FOOD_CONFIG=dict(nutrition_cap=120.,stored_food_cap=6,food_restore=18.,distance_cap=16.,
                 bank_weight=.5,access_weight=1.,distance='observed Manhattan minus interaction radius1; no hidden routes')
REWARD_CONFIG={**deepcopy(ORIGINAL_CONFIG), 'version':'survival_supply_margin_v2',
               'food_potential':deepcopy(FOOD_CONFIG),'reserve_risk':dict(threshold=40.,cost_per_tick_per_need=.0035)}
READINESS_NAMES=tuple(f'{kind}_{field}' for kind in ('berry','mushroom')
    for field in ('available_count','ready_manhattan_distance','ready_dx','ready_dy','next_regrowth','gather_available'))


def food_potential(obs,dead=False):
    validate_observation_identity(obs)
    if type(dead) is not bool:raise ValueError('Boolean terminal flag required')
    p=obs['player'];hunger=_number(p['hunger'],'hunger',0,100)
    food=obs['inventory']['berries']
    if type(food) is not int or food<0:raise ValueError('Nonnegative observed food inventory required')
    if dead:return 0.
    nutrition=min(1.,(hunger+18*min(food,6))/120.)
    distance=16.
    for item in obs['nearby']:
        if item['kind'] not in ('berry','mushroom') or not item.get('available',True):continue
        dx=_number(item['x'],'food x')-_number(p['x'],'player x')
        dy=_number(item['y'],'food y')-_number(p['y'],'player y')
        distance=min(distance,max(0.,abs(dx)+abs(dy)-1))
    return .5*nutrition+1.-(1.-nutrition)*distance/16.


def living_utility(snapshot):
    if type(snapshot['dead']) is not bool:raise ValueError('Boolean death flag required')
    hunger=_number(snapshot['hunger'],'hunger',0,100);thirst=_number(snapshot['thirst'],'thirst',0,100)
    if snapshot['dead']:return 0.
    return .01-.0035*sum(max(0.,1.-v/40.)**2 for v in (hunger,thirst))


class SupplyReward:
    def __init__(self,initial):
        self.base=SurvivalReward(initial)
        self.previous_food=food_potential(initial)

    def score(self,transition,next_observation):
        # Let the unchanged authoritative validator reject broken/closed traces.
        result=self.base.score(transition,next_observation)
        after_food=food_potential(next_observation,transition['after']['dead'])
        gamma=REWARD_CONFIG['gamma_tick']
        shaped=gamma**transition['elapsed_ticks']*after_food-self.previous_food
        risk=sum(gamma**i*(living_utility(row)-(.01 if not row['dead'] else 0.))
                 for i,row in enumerate(transition['trace']))
        self.previous_food=after_food
        result['reward']+=shaped+risk
        result['components'].update(food_potential=shaped,need_risk=risk)
        result['config']=deepcopy(REWARD_CONFIG)
        return result


def reward_config(config):
    variant=config.get('reward_variant','legacy')
    if variant not in ('legacy','supply_margin_v2'):raise ValueError('Unknown registered reward variant')
    return deepcopy(REWARD_CONFIG if variant=='supply_margin_v2' else ORIGINAL_CONFIG)


def create_reward(initial,config):
    reward_config(config)
    return SupplyReward(initial) if config.get('reward_variant','legacy')=='supply_margin_v2' else SurvivalReward(initial)


def readiness_features(prepared):
    """Twelve factual critic inputs from the same positive actor-state whitelist."""
    landmarks=prepared['semantic_state']['地标'];result=[]
    for kind in ('浆果','蘑菇'):
        items=[x for x in landmarks if x['种类']==kind]
        ready=[x for x in items if x['可用']]
        positions=[_position_xy(x['位置']) for x in ready]
        nearest=min(positions,key=lambda xy:(abs(xy[0])+abs(xy[1]),xy)) if positions else None
        regrowth=min([x['再生还需刻数'] for x in items if not x['可用']] or [90])
        result += [min(len(ready),5)/5.,min(abs(nearest[0])+abs(nearest[1]),96)/96. if nearest else 1.,
                   nearest[0]/96. if nearest else 0.,nearest[1]/96. if nearest else 0.,
                   min(regrowth,90)/90.,float(any(x['采集目标'] for x in ready))]
    if len(result)!=12 or not all(math.isfinite(x) for x in result):raise ValueError('Invalid critic readiness facts')
    return result


def expand_critic(critic):
    """Preserve the warm96-D value function; zero-init new factual columns."""
    import torch
    old=critic[0]
    if tuple(old.weight.shape)!=(128,96):raise ValueError('Only compatible96-D critic may be expanded')
    new=torch.nn.Linear(108,128,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:96].copy_(old.weight);new.bias.copy_(old.bias)
    critic[0]=new

