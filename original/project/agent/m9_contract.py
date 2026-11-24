"""Explicit M9 identity adapter; only validated factual fields reach the actor."""
from copy import deepcopy
import json
import math
from agent import m8s3_contract as old
from agent.m8s3_supply import food_potential, living_utility

VERSION="m9_research_v1"
MAPS=("finite","infinite_freeze","infinite_global")
STATES=("legacy","reserves","compact")

def validate(obs):
    if (obs.get("profile")!="m9" or obs.get("world_version")!=7
        or obs.get("environment_version")!="m9_regions_v1"
        or obs.get("map_mode") not in MAPS or obs.get("navigation_steps") not in (1,4)
        or obs.get("balance_sha256")!=old.balance_identity()["balance_sha256"]):
        raise ValueError("Unregistered M9 observation identity")

def legacy_facts(obs):
    # This adapter reuses the old positive field allowlist, not the old world identity.
    validate(obs)
    result=deepcopy(obs)
    result.update(old.balance_identity())
    return result

def encode(obs,state_variant="legacy"):
    if state_variant not in STATES:raise ValueError("Unknown factual representation")
    row=old.encode(legacy_facts(obs))
    rule=row["messages"][0]["content"]
    if obs["navigation_steps"]==4:
        rule=rule.replace("前往地标也只避障走一格，不自动互动","前往由你选定的地标最多连续避障走4格；到达即停，不自动互动，每格仍消耗一刻")
        rule=rule.replace("移动、前往、等待各1刻","移动、等待各1刻；前往每格1刻")
    if obs["map_mode"]!="finite":
        rule+="地图按区生成，旧外边界可跨越；离开区域"+("冻结资源再生时钟，返回不重刷。" if obs["map_mode"]=="infinite_freeze" else "资源仍按世界时间再生。")
    row["messages"][0]["content"]=rule
    if state_variant=="compact":
        state=row["semantic_state"]
        body={"身体":state["身体"],"背包":state["背包"],"时间":state["时间"],"地形":state["地形"],
              "地标":state["地标"],"营地":state["营地"],"威胁":state["威胁"],"合法移动方向":state["合法移动方向"],
              "合法动作":state["合法动作"],"踏入次数":state["走过的位置"],"最近动作":state["最近动作"]}
        row["messages"][1]["content"]=json.dumps(body,ensure_ascii=False,separators=(",",":"))
    elif state_variant=="reserves":
        p=obs["player"];food=obs["inventory"]["berries"]
        facts=f"不补充时饱食耗尽约{math.ceil(p['hunger']/.2)}刻，水分耗尽约{math.ceil(p['thirst']/.08)}刻；背包食物共可补{18*food}饱食。"
        row["messages"][1]["content"]=facts+"\n"+row["messages"][1]["content"]
    row.update(contract=VERSION,state_variant=state_variant)
    return row

def potential(obs,variant,dead=False):
    validate(obs)
    if variant not in ("supply","bottleneck","survival"):raise ValueError("Unknown reward")
    if dead or variant=="survival":return 0.
    p=obs["player"];food=obs["inventory"]["berries"]
    if variant=="supply":
        mapped=legacy_facts(obs)
        return old._training_water(mapped)+food_potential(mapped)
    nutrition=min(1.,(p["hunger"]+18*min(food,6))/120.)
    water=min(1.,p["thirst"]/80.)
    distances={"food":16.,"water":16.}
    for x in obs["nearby"]:
        if not x.get("available",True):continue
        kind="water" if x["kind"]=="water" else "food" if x["kind"] in ("berry","mushroom") else None
        if kind:
            distance=abs(x["x"]-p["x"])+abs(x["y"]-p["y"])
            if kind=="food":distance=max(0,distance-1)
            distances[kind]=min(distances[kind],distance)
    return (.25*p["health"]/100+.75*min(nutrition,water)+.25*nutrition+.25*water
        +.25*(1-nutrition)*(1-distances["food"]/16)+.25*(1-water)*(1-distances["water"]/16))

class Reward:
    def __init__(self,obs,variant="supply",gamma=.9995):
        if not 0<gamma<1:raise ValueError("Discount must be in (0,1)")
        self.validator=old.SurvivalReward(legacy_facts(obs))
        self.variant,self.gamma=variant,gamma
        self.previous=potential(obs,variant,False)
    def score(self,transition,obs):
        self.validator.score(transition,legacy_facts(obs))
        dt=transition["elapsed_ticks"];dead=transition["after"]["dead"]
        after=potential(obs,self.variant,dead)
        shape=self.gamma**dt*after-self.previous
        if self.variant=="supply":
            shape+=self.gamma**dt*old.potential(transition["after"])-old.potential(transition["before"])
        living=sum(self.gamma**i*(living_utility(z) if self.variant=="supply" else .01*int(not z["dead"]))
                   for i,z in enumerate(transition["trace"]))
        self.previous=after
        return dict(reward=living+shape,discount=self.gamma**dt,elapsed_ticks=dt,
                    components=dict(survival=living,potential=shape),variant=self.variant,gamma_tick=self.gamma)

