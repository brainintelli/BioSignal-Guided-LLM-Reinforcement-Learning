"""Reuse the two existing zero-MEA, R1-only references. No new model or new arm is trained."""
import sys
import audit_r2_beta_clock as a

if __name__=='__main__':
    games,cov,_,_=a.load_cells({'22021':['masked'],'22554':['pure']},[0])
    if sys.argv[-1]=='export':
        req={'unique':{}}
        for ch,b,arm,s,w,key,rows,*_ in games:
            if ch=='22021':
                value=[w,[{k:z[k] for k in ('action','context','after','r1')} for z in rows]]
                assert req['unique'].setdefault(key,value)==value
        a.gzsave(a.CACHE/'22021-reference-index.json.gz',req)
        print(len(req['unique']),'reference input sequences')
    else:
        assert all(x['complete_seeds']==16 for x in cov)
        data,_=a.audit(games,a.runreplay(games,'22021-reference-clock.json.gz'),False)
        a.csvout(a.OUT/'r1_reference_worlds.csv',data)
        print(len(data),'reference games audited')
