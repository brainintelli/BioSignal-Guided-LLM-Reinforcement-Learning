import json,pathlib,subprocess,sys
from mea_state2_r2.integrity import verify_assets,check_baseline_identity,baseline_padded_critic_hash
from mea_state2_r2.policy import new_policy
from mea_state2_r2.data import Dataset
from tools.mea_shuffle_train import state
r=pathlib.Path(__file__).parent
assets=verify_assets(r)
d=Dataset(r/'data');p=new_policy(r/'model_config.json','A0',0,d.normalization_id)
check_baseline_identity(state(p),baseline_padded_critic_hash(p))
result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r/'project/standalone/mea_state2_r2/tests')],capture_output=True,text=True)
(r/'unit_tests.log').write_text(result.stdout+result.stderr)
if result.returncode:raise RuntimeError('Unit tests failed')
(r/'preflight.json').write_text(json.dumps({'passed':True,'assets':assets,'baseline':state(p),'baseline_padded_critic_hash':baseline_padded_critic_hash(p)}))
print('PREFLIGHT PASSED',flush=True)
