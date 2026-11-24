"""Native scripted engineering replay, no model efficacy outcomes."""
import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent;project=r/'project';sys.path.insert(0,str(project))
from agent.mea_shuffle import PIN,PROTOCOL,cell_config,training_recording
from tools.mea_ablation import episode,replay_episode
godot=Path(sys.argv[1]);out=r/'native_check';out.mkdir(exist_ok=False)
checks=[]
for arm in PROTOCOL['arms']:
    cfg=cell_config(0,arm);data=training_recording(r/'data',PIN,cfg)
    rows,boot,metrics=episode(None,data,cfg['policy_arm'],list(range(5)),cfg['world_seed'],cfg['action_seed'],out/arm,godot,project)
    replay_episode(rows,boot,metrics,data,cfg['policy_arm'],list(range(5)),cfg['world_seed'],godot,project)
    if not all(x['reward']==x['r1_score']['reward'] and x['reward_record']['r2_enabled'] is False for x in rows):raise ValueError('Reward guard')
    checks.append({'arm':arm,'rows':len(rows),'first_source_window':rows[0]['snapshot']['window_index'] if 'window_index' in rows[0]['snapshot'] else data.order[0]})
(r/'native_check_receipt.json').write_text(json.dumps({'passed':True,'scripted_only':True,'model_outcomes':False,'checks':checks},indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'scripted_only':True,'checks':checks}))
