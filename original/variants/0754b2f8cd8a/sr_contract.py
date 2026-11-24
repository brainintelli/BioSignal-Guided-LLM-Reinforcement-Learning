"""Opt-in matched state representation, explicitly a filtered offline source."""
from agent import m9_contract
from agent.m8s3_supply import readiness_features
from sr_data import digest,canonical
from sr_protocol import ARMS,STATE_ARMS,STATE_FEATURES,PROTOCOL
CONTRACT='mea_22554_experimental_recording_v1'
def context_id(obs):return digest(m9_contract.encode(obs,'reserves'))
def encode(obs,snapshot,arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    s=snapshot
    if s.get('snapshot_id')!=digest({k:v for k,v in s.items() if k!='snapshot_id'}):raise ValueError('snapshot changed')
    if s.get('context_id')!=context_id(obs) or s.get('source_sha256')!=PROTOCOL['source_sha256']:raise ValueError('snapshot context/source differs')
    if s.get('source_full_qc_passed') is not False or s.get('coverage_basis')!='filtered_finalized_h5_subset' or s.get('association')!='prerecorded_action_independent':raise ValueError('source qualification changed')
    row=m9_contract.encode(obs,'reserves');state=list(s['state_values']) if arm in STATE_ARMS else [0.]*4
    if len(state)!=4 or any(not isinstance(x,(int,float)) or not -5<=x<=5 for x in state):raise ValueError('state vector invalid')
    if arm in STATE_ARMS:
        row['messages'][1]['content']+='\n离线神经观测（校准标准化）：'+canonical({'z_'+k:round(v,6) for k,v in zip(STATE_FEATURES,state)})+'；无刺激历史记录，与本次游戏动作无因果对应。'
    row.update(contract=CONTRACT,base_contract=m9_contract.VERSION,critic_input=row['features']+readiness_features(row)+state,
        state_values=state,snapshot_id=s['snapshot_id'])
    return row
