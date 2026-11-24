"""P0 guard: R2 is uncalibrated telemetry; total_reward is exactly R1."""
from agent.mea_protocol import identifier, number


def _metrics(values):
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError('Metric mapping required')
    return {identifier(key, 'metric name'): number(value, key) for key, value in values.items()}


def record_reward(r1, *, alpha=1., r2_raw=None, neural_metrics=None, candidate_scores=None):
    if number(alpha, 'alpha') != 1:
        raise ValueError('P0 requires alpha=1; R2 cannot affect reward')
    number(r1, 'r1')
    if r2_raw is not None:
        number(r2_raw, 'r2_raw')
    return dict(alpha=1., r1=r1, total_reward=r1, r2_raw=r2_raw, r2_enabled=False, r2_valid=False,
                r2_status='uncalibrated_record_only', neural_metrics=_metrics(neural_metrics),
                candidate_scores=_metrics(candidate_scores))
