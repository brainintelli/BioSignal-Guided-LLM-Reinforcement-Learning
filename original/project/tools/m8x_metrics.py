"""Unchanged survival gates, mapped to the new three-checkpoint exposure budget."""
from tools.m8b_metrics import (HORIZON, summarize, matrix, binomial_interval,
    lower_tail_mean, compare_test, qualify_selection as legacy_qualify_selection)

CHECKPOINTS=(32768,65536,98304)


def qualify_selection(candidates,parent_episodes):
    if set(candidates)!=set(CHECKPOINTS):
        raise ValueError('All three exploration checkpoints required')
    # The ranking/statistics are unchanged; the exposure labels are doubled.
    # This mapping preserves ordering and never alters any episode or metric.
    result=legacy_qualify_selection({n//2:rows for n,rows in candidates.items()},parent_episodes)
    result['selected_decisions']*=2
    result['summaries']={n*2:value for n,value in result['summaries'].items()}
    return result
