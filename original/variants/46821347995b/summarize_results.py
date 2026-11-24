"""Descriptive results only; registered primary inference remains unchanged."""
import collections,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence/experiment01';ARMS=('baseline','state','reward','both','shuffled')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mean(xs):return statistics.mean(xs)
def main():
    assert read(R/'download_verification.json')['passed'];analysis=read(E/'analysis.json');result=dict(arms={},blocks=[],runtime={})
    for arm in ARMS:
        reports=[read(E/f'b{b:02d}_{arm}/report.json') for b in range(8)];summaries={}
        for phase in ('pre','post'):
            rows=[x for r in reports for x in r[phase]];actions=collections.Counter()
            for b in range(8):
                for j in range(2):
                    for line in (E/f'b{b:02d}_{arm}'/(phase+str(j))/'rollout.jsonl').read_text(encoding='utf-8').splitlines():actions[json.loads(line)['action']['verb']]+=1
            summaries[phase]=dict(discounted_r1=mean(x['discounted_r1'] for x in rows),raw_r1=mean(x['r1_sum'] for x in rows),
                ticks=mean(x['ticks'] for x in rows),decisions=mean(x['decisions'] for x in rows),deaths=sum(x['natural_terminals'] for x in rows),worlds=16,
                final_health=mean(x['final_observation']['player']['health'] for x in rows),final_hunger=mean(x['final_observation']['player']['hunger'] for x in rows),
                final_thirst=mean(x['final_observation']['player']['thirst'] for x in rows),
                final_inventory={k:mean(x['final_observation']['inventory'].get(k,0) for x in rows) for k in ('berries','wood','stone')},
                action_counts=dict(actions),wait_fraction=actions['wait']/sum(actions.values()),max_input_tokens=max(x['max_input_tokens'] for x in rows))
            assert abs(summaries[phase]['discounted_r1']-analysis['arms'][arm][phase])<1e-12
        summaries['gain']=summaries['post']['discounted_r1']-summaries['pre']['discounted_r1']
        summaries['train']=dict(original_r1_mean_sum=mean(r['train']['r1_sum'] for r in reports),raw_aux_mean_sum=mean(r['train']['raw_aux_sum'] for r in reports),
            optimized_mean_sum=mean(r['train']['optimized_reward_sum'] for r in reports),actor_steps=[r['update']['metrics']['optimizer_steps'] for r in reports],
            critic_steps=[r['update']['critic_steps'] for r in reports],collector_mean_seconds=mean(r['seconds'] for r in reports))
        result['arms'][arm]=summaries
        for b,r in enumerate(reports):result['blocks'].append(dict(block=b,arm=arm,pre=mean(x['discounted_r1'] for x in r['pre']),post=mean(x['discounted_r1'] for x in r['post'])))
    reports=[read(E/f'b{b:02d}_{a}/report.json') for b in range(8) for a in ARMS];resources=[x for r in reports for x in r['resources']]
    result['runtime']=dict(seconds=read(E/'driver_status.json')['seconds'],max_peak_allocated_gib=max(x['peak_allocated'] for x in resources)/1024**3,
        max_peak_reserved_gib=max(x['peak_reserved'] for x in resources)/1024**3,max_process_peak_rss_gib=max(x['process_peak_rss'] for x in resources)/1024**3,
        max_phase_end_rss_gib=max(x['rss_after_release'] for x in resources)/1024**3,
        actor_steps_total=sum(r['update']['metrics']['optimizer_steps'] for r in reports),critic_steps_total=sum(r['update']['critic_steps'] for r in reports))
    with (R/'descriptive_results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
