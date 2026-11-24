"""Shared terminal/cutoff logic, independent of model and simulator."""
import math

def bootstrap(rows,next_observation,prepare,value):
    assert rows
    if rows[-1]['terminal']:
        return 0.,None
    prepared=prepare(next_observation);result=float(value(prepared))
    assert math.isfinite(result)
    return result,prepared

def validate_tail(rows,expected_length):
    assert expected_length>=2 and len(rows)==expected_length
    assert all(not r['terminal'] for r in rows[:-1]) and rows[-1]['terminal']
