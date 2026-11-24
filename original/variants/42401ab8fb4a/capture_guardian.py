"""Owned one-shot acquisition, audit and sealing on the recording host."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

BASE=Path('/home/researcher/mea-acquisition')
SOFTWARE=BASE/'long_software_20261116_v1'
PACKAGE=BASE/'long_delivery_20261116'
RUN=PACKAGE/'long_20261116_run01'
PYTHON='/pub/soft/MaxLab/python/bin/python3'
STATUS=BASE/'long_capture_status_20261116.json'
state={'phase':'starting','stimulation':False,'alpha':1,'r2_enabled':False,'realtime_training':False}

def persist():
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=STATUS.with_suffix('.partial.json');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATUS)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for b in iter(lambda:handle.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    PACKAGE.mkdir(exist_ok=False)
    persist()
    watcher_code="from pathlib import Path; from tools.mea_progress_watch import watch; import json; r=watch(Path(%r),Path(%r),1830); print(json.dumps(r)); raise SystemExit(0 if r['passed'] else 2)"%(str(RUN/'events.jsonl'),str(PACKAGE/'live_watch.jsonl'))
    with (PACKAGE/'watcher.log').open('x') as log:
        watcher=subprocess.Popen([PYTHON,'-s','-c',watcher_code],cwd=SOFTWARE,stdout=log,stderr=subprocess.STDOUT)
    cmd=[PYTHON,'-s',str(SOFTWARE/'tools/mea_p1_acquire.py'),'--output',str(RUN),
         '--collector',str(SOFTWARE/'mea_timing_capture_long'),'--seconds','1800','--legacy-pid','1913926','--chip-id','22021']
    with (PACKAGE/'coordinator.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=SOFTWARE,stdout=log,stderr=subprocess.STDOUT)
    state.update(phase='acquiring',coordinator_pid=child.pid,watcher_pid=watcher.pid);persist()
    try:code=child.wait(timeout=1860)
    except subprocess.TimeoutExpired:
        child.terminate()
        state['timeout_graceful_stop']=True;persist()
        try:code=child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            state['phase']='cleanup_requires_inspection';persist();return 2
    state.update(coordinator_exit=code,phase='closed' if code==0 else 'capture_failed');persist()
    try:state['watcher_exit']=watcher.wait(timeout=25)
    except subprocess.TimeoutExpired:
        watcher.terminate();state['watcher_exit']=watcher.wait(timeout=10)
    persist()
    if code:return 2
    for name,module in [('event_audit','tools.mea_p1_audit'),('timing_audit','tools.mea_timing_audit')]:
        state['phase']=name;persist()
        with (PACKAGE/(name+'.log')).open('x') as log:
            result=subprocess.run([PYTHON,'-s','-m',module,str(RUN)],cwd=SOFTWARE,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        state[name+'_exit']=result.returncode;persist()
        if name=='event_audit' and result.returncode:
            state['phase']='audit_failed';persist();return 2
    audit=json.loads((RUN/'audit.json').read_text());acq=json.loads((RUN/'acquisition.json').read_text())
    assert audit['offline_integrity_passed'] and acq['success'] and acq['legacy_restored']
    manifest={'accepted_run':RUN.name,'schema':'mea.sealed.recording.v1',
        'files':{p.relative_to(PACKAGE).as_posix():{'sha256':sha(p),'size':p.stat().st_size}
                 for p in sorted(PACKAGE.rglob('*')) if p.is_file()}}
    (PACKAGE/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
    state.update(phase='sealed',manifest_sha256=sha(PACKAGE/'manifest.json'),
        raw_frames=audit['raw_frame_count'],seconds=audit['raw_duration_seconds'],matched_events=audit['matched_events'],
        restored_legacy_pid=acq['restored_legacy_pid'],maxlab_pid=acq['maxlab_pid']);persist()
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        state.update(phase='guardian_failed',error=repr(e));persist();raise
