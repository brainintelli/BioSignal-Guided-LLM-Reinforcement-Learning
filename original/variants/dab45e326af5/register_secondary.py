"""Fix one lower-tail endpoint from the previous control distribution, before inspecting new outcomes."""
from pathlib import Path
import csv,hashlib,json,math
H=Path(__file__).resolve().parent
def main():
    prev=H.parent/'paired.csv'; prior=list(csv.DictReader(prev.open(encoding='utf-8-sig')))
    vals=sorted(int(v['masked_new_cells']) for v in prior)
    rank=math.ceil(.25*len(vals));cut=vals[rank-1]
    spec=dict(endpoint='fraction of games finishing at or below the previous masked lower-quartile discovery count',
        cutoff=cut,rule='nearest-rank 25th percentile inclusive; prior masked arm only',rank=rank,n_prior=len(vals),
        comparison='masked low-exploration rate minus ordered rate; positive favors chronological MEA',
        confirmation='estimate on new maps with untouched threshold; 95% paired crossed interval, exploratory secondary only',
        prior_pairs_sha=hashlib.sha256(prev.read_bytes()).hexdigest(),
        design_sha=hashlib.sha256((H/'design.json').read_bytes()).hexdigest(),
        chronology='added after dispatch, before reading any new episode outcomes; only the original control distribution sets the threshold')
    with (H/'lower-tail-plan.json').open('x',encoding='utf8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print('old control Q1 cutoff',cut,'rank',rank,'of',len(vals))
if __name__=='__main__':main()
