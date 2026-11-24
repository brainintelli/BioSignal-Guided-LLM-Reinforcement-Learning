"""Scoped mechanical fork of the accepted trainer; original files remain sealed."""
from pathlib import Path
r=Path(__file__).resolve().parent/'project'
text=(r/'tools/mea_stim_train.py').read_text(encoding='utf-8')
text=text.replace('agent.mea_stim_offline','agent.mea_shuffle').replace('mea_stim_train.py','mea_shuffle_train.py')
old="return StimRecording(source,pin=pin,phase='after' if ablation['arm']=='after' else 'before',split='train')"
assert text.count(old)==1
text=text.replace(old,"from agent.mea_shuffle import training_recording\n    return training_recording(source,pin,ablation)")
(r/'tools/mea_shuffle_train.py').write_text(text,encoding='utf-8')
text=(r/'tools/mea_stim_campaign.py').read_text(encoding='utf-8')
text=text.replace('agent.mea_stim_offline','agent.mea_shuffle').replace('mea_stim_train','mea_shuffle_train').replace('mea_stim_campaign','mea_shuffle_campaign')
old="cells.append(dict(block=b,arm=a,**{p:{c:e['metrics'][p][c]['discounted_r1'] for c in ('before','after')} for p in ('pre','post')}))"
assert text.count(old)==1
text=text.replace(old,"cells.append(dict(block=b,arm=a,**{p:{c:e['metrics'][p][c] for c in ('before','after')} for p in ('pre','post')}))")
text=text.replace("for b in range(8):_same(indexed[b,'before']['pre'],indexed[b,'after']['pre'],'identical common initial evaluation')", "for b in range(8):\n            for arm in ('after','shuffled_after'):_same(indexed[b,'before']['pre'],indexed[b,arm]['pre'],'identical common initial evaluation')")
text=text.replace('planned_cells=24','planned_cells=32').replace('orders[b%6]','orders[b%len(orders)]')
text=text.replace("print(canonical(result),flush=True)","print(canonical(result),flush=True)\n    if isinstance(result,dict) and result.get('passed') is False:raise SystemExit(1)")
(r/'tools/mea_shuffle_campaign.py').write_text(text,encoding='utf-8')
print('Forked accepted trainer/campaign; original source files unchanged.')
