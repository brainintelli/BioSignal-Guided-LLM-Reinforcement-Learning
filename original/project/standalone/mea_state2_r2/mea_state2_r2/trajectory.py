"""Time-aware trajectory boundaries and independent advantage audit."""
import math
from agent.rl_contract import gae


def finish(rows, bootstrap, gamma, lam):
    if not rows or not math.isfinite(bootstrap):
        raise ValueError('Finite nonempty trajectory required')
    for i, row in enumerate(rows):
        if type(row['terminal']) is not bool or type(row['duration']) is not int or row['duration'] < 1:
            raise ValueError('Invalid terminal/duration')
        row['boundary'] = row['terminal'] or i == len(rows) - 1
        row['next_value'] = 0. if row['terminal'] else rows[i+1]['value'] if i+1 < len(rows) else bootstrap
    advantages, returns = gae(*[[r[k] for r in rows] for k in
        ('reward','value','next_value','duration','terminal','boundary')], gamma, lam)
    for row, advantage, target in zip(rows, advantages, returns):
        row.update(advantage=advantage, **{'return': target})
    audit(rows, bootstrap, gamma, lam)


def audit(rows, bootstrap, gamma, lam):
    carry = 0.
    for i in reversed(range(len(rows))):
        row = rows[i]
        boundary = row['terminal'] or i == len(rows)-1
        nxt = 0. if row['terminal'] else rows[i+1]['value'] if i+1 < len(rows) else bootstrap
        discount = gamma ** row['duration']
        carry = row['reward'] + discount*nxt - row['value'] + (0. if boundary else discount*lam**row['duration']*carry)
        if (row['boundary'] != boundary or row['next_value'] != nxt
                or not all(math.isfinite(row[k]) for k in ('reward','value','advantage','return'))
                or abs(row['advantage']-carry) > 1e-10 or abs(row['return']-(carry+row['value'])) > 1e-10):
            raise ValueError('Trajectory audit mismatch')
