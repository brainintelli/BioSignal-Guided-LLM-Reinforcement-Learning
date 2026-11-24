"""No equipment transport, no optimizer update. Exercise the new input with archived frames."""
import json
from pathlib import Path
import sys


def main():
    import torch
    from agent.m9_contract import encode
    from mea_online.m9_runtime import M9Runtime, digest, project_observation
    from livepair.protocol import PROTOCOL,snapshot
    here=Path(__file__).resolve().parent
    src=json.loads((here/'historical_replay.json').read_text())
    cfg=json.loads((here/'server.json').read_text())
    norm=src['norm']
    # Session remains explicitly in the equipment-free runtime mode for this probe.
    config=dict(source='simulation',stimulation=False,r2_enabled=False,devices=list(range(8)),
                batch_size=16,binding_sha256=src['bid'],normalization_id=digest(norm),normalization=norm,
                seed=740001,probe_reference='rtx4090-20261116',**{k:cfg[k] for k in ('model_paths','project','godot')})
    rt=M9Runtime(config,here/'technical_model_check')
    try:
        obs=rt.reset(740001)
        ctx=digest(encode(obs,'reserves'))
        s=snapshot(src['windows'][0],config,ctx,'archived-input-check')
        row=project_observation(obs,s,dict(config,experiment=PROTOCOL))
        policy=rt.core.policy
        ids=policy._tokens(row['messages'])
        row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={})
        row['prepared_sha256']=digest(row)
        before=rt.state()
        torch.manual_seed(740001)
        sample=rt.act(row)
        step=rt.step(sample['action'])
        if row['critic_input'][-4:]!=s['normalized'] or before!=rt.state():
            raise ValueError('Projected input or frozen weights differ')
        result=dict(devices=config['devices'],model_state=before,real_godot=True,archive_chip=src['chip'],
                    archive_window=s['window']['id'],normalized=s['normalized'],critic_width=len(row['critic_input']),
                    action=sample['action'],native_r1=step['reward'],duration=step['duration'],
                    prepared_hash=row['prepared_sha256'],reload_probe=rt.core.policy.migration_probe,
                    real_stimulation=False,optimizer_updates=0)
    finally:
        rt.close()
    (here/'model_check.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
