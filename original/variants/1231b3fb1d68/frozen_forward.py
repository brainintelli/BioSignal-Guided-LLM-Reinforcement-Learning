import json,math,sys,hashlib,statistics
from pathlib import Path
import torch
from agent.m8s3_contract import VERBS
from agent.mea_contract import observation_key
from mea_state2_r2.data import Dataset
from mea_state2_r2.features import digest,extract,transform
from mea_state2_r2.policy import new_policy,load_checked
from tools.mea_shuffle_train import state

where=Path(__file__).resolve().parent
run=Path('/home/researcher/mea-state2-r2-20261118/run01')
inp=json.loads((where/'inference-input.json').read_text())
data=Dataset(run/'data')
obs=inp['obs'];context=observation_key(obs)
ckpt=run/'experiment01/b00_C/train/round_03/checkpoint'
manifest=json.loads((ckpt/'manifest.json').read_text())
frozen=new_policy(run/'model_config.json','C',0,data.normalization_id)
load_checked(frozen,ckpt)
frozen.model.eval();frozen.critic.eval()
start_model=state(frozen)
print('Frozen trained checkpoint loaded', flush=True)
r=[];evidence={}
fixed=[]
with torch.inference_mode():
 for row in inp['samples']+[inp['samples'][0]]:
  phase,i=row['phase'],row['i'];raw=row['snapshot']
  snap=data.snapshot(i,phase,context,'C',0,'eval')
  assert snap['window_start']==raw['window_start'] and snap['window_end']==raw['window_end'] and snap['events_sha256']==raw['events_sha256']
  record=data.recordings[phase].rows[snap['window_index']]
  assert row['frame_channel_pairs']==[[f,c] for f,c,_ in record['events']]
  v=extract([(f,c,0.) for f,c in row['frame_channel_pairs']],snap['window_start'],snap['window_end'],tuple(c for c,_ in data.recordings[phase].mapping),20000)
  assert list(v)==snap['features'] and list(transform(v,data.normalization))==snap['normalized']==raw['normalized']
  p=frozen.prepare([obs],[snap])[0]
  assert p['critic_input'][-4:]==snap['normalized']
  prob=frozen.verb_distribution([p]).probs[0].detach().float().cpu().tolist()
  val=frozen.values([p]).item()
  assert len(prob)==len(VERBS) and math.isclose(sum(prob),1.,abs_tol=1e-6)
  out=dict(phase=phase,index=i,window=snap['window_index'],window_start=snap['window_start'],window_end=snap['window_end'],spikes=row['spikes'],normalized=snap['normalized'],verbs_prob=prob,greedy_verb=VERBS[max(range(len(prob)),key=prob.__getitem__)],value=val,context_id=context,snapshot_id=snap['snapshot_id'])
  if len(r)==32:
   assert out==r[0];break
  if i==0:
   archived=inp['archived_first'][phase]
   with (run/'experiment01/b00_C/eval'/phase/'rollout.jsonl').open() as f:ref=json.loads(next(f))
   assert snap==ref['snapshot']
   idx=archived['verb']; difference=abs(math.log(prob[idx])-archived['verb_logp'])
   evidence[phase]=dict(original_value=archived['value'],value_error=abs(val-archived['value']),original_chosen_verb=VERBS[idx],abs_logprob_error=difference)
  r.append(out)
end=state(frozen)
assert end==start_model and start_model['actor']==manifest['actor_sha256']
before=r[:16];after=r[16:]
d=[sum(abs(a-b) for a,b in zip(x['verbs_prob'],y['verbs_prob']))/2 for x,y in zip(before,after)]
summary=dict(paired_windows=16,total_variation_mean=statistics.mean(d),total_variation_max=max(d),greedy_verb_changed=sum(x['greedy_verb']!=y['greedy_verb'] for x,y in zip(before,after)),greedy_verbs_before=[x['greedy_verb'] for x in before],greedy_verbs_after=[x['greedy_verb'] for x in after],value_mean_before=statistics.mean(x['value'] for x in before),value_mean_after=statistics.mean(x['value'] for x in after),mean_probs_before=[statistics.mean(x['verbs_prob'][j] for x in before) for j in range(len(VERBS))],mean_probs_after=[statistics.mean(x['verbs_prob'][j] for x in after) for j in range(len(VERBS))])
result=dict(archive_chip=inp['chip'],stim_event_frames=inp['stim_event_frames'],excluded_stim_channel=inp['excluded_readout'],stim_electrode=inp['stim_electrode'],independent_channels=inp['independent_channels'],sampling_hz=inp['sample_rate_hz'],after_seconds_from_last_tail=[(after[0]['window_start']-inp['last_pulse_end_frame'])/20000,(after[-1]['window_end']-inp['last_pulse_end_frame'])/20000],raw_sha256=inp['source_raw_sha256'],game_context_sha256=context,checkpoint=str(ckpt),model_state=start_model,device=frozen.device_name.__str__(),optimizer_updates=0,real_stimuli_this_run=0,rewards_issued=0,same_input_repeat_exact=True,archived_first_replay=evidence,normalization_id=data.normalization_id,verbs=list(VERBS),summary=summary,rows=r)
with (where/'forward-result.json').open('x') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2)
print(json.dumps({'summary':summary,'archived_first_replay':evidence}),flush=True)
