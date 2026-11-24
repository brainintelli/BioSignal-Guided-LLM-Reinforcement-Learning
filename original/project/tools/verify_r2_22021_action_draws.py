"""Independent GPU replay of saved action selection using the frozen actor probability vectors."""
import gzip,json
from pathlib import Path
import torch
from torch.distributions import Categorical
root=Path.home()/'r2-evidence-20261120'
def load(name):
    with gzip.open(root/name,'rt',encoding='utf8') as f:return json.load(f)
p=load('22021-distributions.json.gz');draws=load('22021-action-choices.json.gz')
assert p.keys()==draws.keys()
matched=n=0; wrong=[]
with torch.no_grad():
    for condition,decisions in draws.items():
        for world,j,ctx,act in decisions:
            probs=p[condition][f'{j}:{ctx}'];dist=Categorical(probs=torch.tensor([probs],dtype=torch.float32,device='cuda:0'))
            torch.manual_seed(world*10+j)
            verb=dist.sample().item();matched+=int(verb==act);n+=1
            if verb!=act and len(wrong)<3:wrong.append([condition,world,j,verb,act])
out={'frozen_models':len(draws),'decisions':n,'matched_logged_verbs':matched,'mismatch_examples':wrong}
print(json.dumps(out));(root/'action-draw-check.json').write_text(json.dumps(out))
assert matched==n==36864
