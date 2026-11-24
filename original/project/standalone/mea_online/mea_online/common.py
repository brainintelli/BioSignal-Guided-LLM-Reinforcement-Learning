"""Pure helpers; importing this module never opens devices or models."""
import hashlib
import json
import math
import os
from pathlib import Path

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def number(value,name):
    if type(value) not in (float,int) or not math.isfinite(value):raise ValueError(name+': finite number required')
    return value
def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(name+': integer required')
    return value
def clone(value):return json.loads(canonical(value))
def read(path):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('Duplicate JSON key '+k)
            d[k]=v
        return d
    def invalid(s):raise ValueError('Invalid JSON constant '+s)
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=invalid)
def write(path,value,exclusive=True):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
def append(path,value):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
