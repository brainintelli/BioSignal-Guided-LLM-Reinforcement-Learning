"""Exercise the released GPU HTTP service with two real Godot decisions.

This is a transport smoke test, never a survival assessment or training sample.
Negative requests must be rejected without drawing replacement actions.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.m6_live_session import STOCHASTIC, live_seed
from agent.broker import validate_action
from agent.rl_env import GodotEnv


def request(origin, path, observation=None, context=None):
    headers = {'Content-Type': 'application/json'}
    if context:
        headers.update({'X-Agent-Session': context['session_id'],
                        'X-Agent-Episode': str(context['episode']),
                        'X-World-Seed': str(context['world_seed'])})
    payload = None if observation is None else json.dumps(observation).encode()
    req = urllib.request.Request(origin + path, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--godot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.url)
    require(parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
            and not parsed.username and not parsed.query and not parsed.fragment
            and parsed.path in ('', '/'), 'Only the local SSH-forwarded origin is permitted')
    output = Path(args.output)
    require(not output.exists(), 'Refusing to replace prior evidence')
    origin = args.url.rstrip('/')
    status, health = request(origin, '/health')
    require(status == 200 and health['survival_gate_passed'] is True
            and health['decision_ready'] is True and health['decoding'] == STOCHASTIC
            and health['temperature'] == 1 and health['transport_context_required'] is True
            and health['scripted_fallback'] is False and health['task_goals'] is False,
            'Endpoint has not passed the required stochastic release gate')
    context = {'session_id': 'm6-link-probe-' + uuid.uuid4().hex,
               'episode': 0, 'world_seed': 2304}
    checks = []
    records = []
    with GodotEnv(args.godot, str(ROOT)) as environment:
        observation = environment.reset(context['world_seed'])['observation']
        status, error = request(origin, '/decide', observation)
        require(status == 400, 'Missing transport context was not rejected')
        checks.append({'check': 'missing_context_rejected', 'status': status, 'error': error})
        first_observation = copy.deepcopy(observation)
        previous = None
        for _ in range(2):
            status, decision = request(origin, '/decide', observation, context)
            require(status == 200 and decision['source'] == 'model'
                    and decision['raw_valid'] is True and decision['tick'] == observation['tick']
                    and decision['actor_sha256'] == health['actor_sha256']
                    and decision['decoding'] == STOCHASTIC and decision['agent_context'] == context
                    and decision['rng_seed'] == str(live_seed(context)), 'Model identity/context mismatch')
            if previous:
                require(previous['rng_after_sha256'] == decision['rng_before_sha256'], 'RNG chain changed')
            status, retry = request(origin, '/decide', observation, context)
            require(status == 200 and retry == decision, 'Retry drew a replacement action')
            changed = copy.deepcopy(observation)
            changed['inventory']['berries'] += 1  # Negative request only; never executed in Godot.
            changed_status, error = request(origin, '/decide', changed, context)
            require(changed_status == 400 and error.get('detail') == 'Changed observation at an already decided tick',
                    'Changed same-tick observation was not rejected by the retry guard')
            status, cached = request(origin, '/decide', observation, context)
            require(status == 200 and cached == decision, 'Rejected retry altered the committed decision')
            action = validate_action(decision['action'], observation)
            transition = environment.step(action)
            require(transition['observation']['tick'] > observation['tick'], 'Real world did not advance')
            records.append({'observation': observation, 'decision': decision,
                            'transition': transition, 'retry_equal': True,
                            'changed_same_tick_status': changed_status, 'changed_same_tick_error': error})
            observation = transition['observation']
            previous = decision
        status, error = request(origin, '/decide', first_observation, context)
        require(status == 400 and error.get('detail') == 'Stale observation', 'Stale tick was not rejected')
        checks.append({'check': 'stale_tick_rejected', 'status': status, 'error': error})
    result = {'passed': True, 'scope': 'HTTP transport and two actual Godot actions; not survival evidence',
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'health': health,
              'agent_context': context, 'checks': checks, 'records': records,
              'executed_model_actions': len(records), 'fallback_actions': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'executed_model_actions': len(records),
                      'actor_sha256': health['actor_sha256'], 'output': str(output)}))


if __name__ == '__main__':
    main()
