import json,sys
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');sys.path.insert(0,str(r/'project'))
from agent.m8x_contract import REWARD_CONFIG,balance_identity
from agent.m8x_runtime import kernel_identity
from agent.m8x_policy_identity import PARENT_ACTOR_SHA256
p=json.loads((r/'model_config.json').read_text());meta=json.loads((Path(p['initial_checkpoint'])/'policy.json').read_text())
expected={'critic_features':97,'reference_actor_sha256':PARENT_ACTOR_SHA256,'physical_forward_shape':[4,1536],**balance_identity(),'reward_config':REWARD_CONFIG,'kernel_identity':kernel_identity()}
diff={k:{'checkpoint':meta.get(k),'current':v} for k,v in expected.items() if meta.get(k)!=v}
print(json.dumps({'differences':diff},indent=2))
