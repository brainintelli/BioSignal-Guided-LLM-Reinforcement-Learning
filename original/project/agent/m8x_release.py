"""Bind an open3 serving policy to the independently rebuilt held-out result."""
import json
from pathlib import Path

from agent.m8x_contract import VERSION as CONTRACT, balance_identity, REWARD_CONFIG
from agent.m8x_identity import BASE_WEIGHT_SHA
from agent.m8x_policy_identity import PARENT_ACTOR_SHA256
from tools.compare_m8x_survival import build_result

LAYOUT = 'fixed_batch4_left_pad_v1'
SHAPE = [4, 1536]


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def rebuild_release(report, project):
    """Dispatch only known protocols, then independently rebuild every gate."""
    protocol = report.get('protocol')
    if protocol != 'm8_exploration_reliable_v1':
        raise ValueError('Unsupported release protocol')
    rebuilt = build_result(report['publication_request'], project)
    if canonical(report) != canonical(rebuilt):
        raise ValueError('Complete exploration-world publication result does not reproduce')
    if (report.get('phase') != 'test' or report.get('decision') != 'accept'
            or report.get('release_authorized') is not True
            or report.get('evidence_complete') is not True):
        raise ValueError('No complete accepted exploration-world held-out release')
    return rebuilt


def verify_release(*, verification, project, checkpoint, model, actor_sha256,
                   policy_metadata, execution_identity):
    project, checkpoint, model = (Path(p).resolve() for p in (project,checkpoint,model))
    report = json.loads(Path(verification).read_text(encoding='utf-8'))
    report = rebuild_release(report, project)
    config = report['config']
    if (Path(config['project']).resolve() != project or Path(config['model']).resolve() != model
            or Path(config['output']).resolve()/report['selected_checkpoint'] != checkpoint
            or Path(report['publication_request']['training_run']).resolve() != checkpoint.parent
            or report['selected_checkpoint'] not in ('checkpoint_0016','checkpoint_0032','checkpoint_0048')
            or config.get('contract') != CONTRACT or config.get('max_tokens') != 1536
            or config.get('temperature') != 1):
        raise ValueError('Serving paths or configuration differ from registered policy')
    world = balance_identity(project)
    for key,value in world.items():
        if policy_metadata.get(key) != value:
            raise ValueError('Loaded policy world differs: '+key)
        if key != 'profile' and report.get(key) != value:
            raise ValueError('Verified report world differs: '+key)
        if key != 'balance_sha256' and config.get(key) != value:
            raise ValueError('Registered world differs: '+key)
    if (report.get('selected_actor_sha256') != actor_sha256
            or report.get('parent_actor_sha256') != PARENT_ACTOR_SHA256
            or canonical(execution_identity) != canonical(report['execution_identity'])
            or execution_identity.get('base_weight_sha256') != BASE_WEIGHT_SHA):
        raise ValueError('Serving actor, parent, base, tokenizer or runtime differs')
    saved = json.loads((checkpoint/'policy.json').read_text(encoding='utf-8'))
    if canonical(saved) != canonical(policy_metadata):
        raise ValueError('Loaded policy metadata differs from selected artifact')
    expected = dict(actor_sha256=actor_sha256,contract=CONTRACT,
        reference_actor_sha256=PARENT_ACTOR_SHA256,inference_layout=LAYOUT,
        physical_forward_shape=SHAPE,max_tokens=1536,task_goals=False,policy_override=False)
    expected.update(reward_config=REWARD_CONFIG,kernel_identity=execution_identity.get('runtime',{}).get('kernel_identity'))
    if not expected['kernel_identity']:raise ValueError('Missing tested GPU implementation identity')
    if any(policy_metadata.get(k) != v for k,v in expected.items()):
        raise ValueError('Loaded policy is not the evaluated exploration representation/layout')
    if policy_metadata.get('task_goals') is not False or policy_metadata.get('policy_override') is not False:
        raise ValueError('Serving policy assistance flags must be false')
    return report
