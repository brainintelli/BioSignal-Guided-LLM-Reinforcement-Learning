"""Bounded, read-only QC subprocess. Never imports a device or controller module.

Run via ``python -m mea_stim_v2.worker --request FILE --response FILE``.
Only closed HDF5 files and JSON requests cross the process boundary. Device
sockets, pipes, handles, and SDK objects are never passed to the child.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .config import Config, digest


class WorkerError(RuntimeError):
    pass


def _stat(path):
    p=Path(path).resolve();s=p.stat()
    if not p.is_file():raise WorkerError('QC source must be a closed regular file')
    return {'path':str(p),'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _load_json(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def _stage(stage):
    if stage in ('preflight','baseline','pilot','recovery'):
        return stage,300. if stage in ('baseline','pilot') else 30.
    match=re.fullmatch(r'pair_(0?[1-9]|10)',str(stage))
    if match:return 'pair',30.
    raise WorkerError('Unknown QC stage: '+str(stage))


def _failed(request,error):
    return {'stage':request['stage'],'status':'failed','errors':[str(error)],
            'source':request['source'],'session_id':request['session_id'],
            'binding_sha256':request['binding_sha256'],'r2_enabled':False,'training_enabled':False}


def _envelope(request,result):
    return {k:request[k] for k in ('request_sha256','config_sha256','session_id','binding_sha256','stage','source','baseline_source')} | {
        'worker_pid':os.getpid(),'result':result,'result_sha256':digest(result)}


def _validate_response(request,response):
    if not isinstance(response,dict):raise WorkerError('Worker result is not an object')
    for key in ('request_sha256','config_sha256','session_id','binding_sha256','stage','source','baseline_source'):
        if response.get(key)!=request[key]:raise WorkerError('Worker response binding mismatch: '+key)
    result=response.get('result')
    if not isinstance(result,dict) or result.get('status') not in ('pass','failed') or not isinstance(result.get('errors'),list):
        raise WorkerError('Invalid worker QC result schema')
    if response.get('result_sha256')!=digest(result):raise WorkerError('Worker result digest mismatch')
    for key in ('stage','source','session_id','binding_sha256'):
        if result.get(key)!=request[key]:raise WorkerError('QC result identity mismatch: '+key)
    if result.get('r2_enabled') is not False or result.get('training_enabled') is not False:
        raise WorkerError('QC result R2/training state not off')
    if result['status']=='pass' and result['errors']:raise WorkerError('Passing QC contains errors')
    if _stat(request['source']['path'])!=request['source']:raise WorkerError('QC source changed during worker run')
    if request['baseline_source'] is not None and _stat(request['baseline_source']['path'])!=request['baseline_source']:
        raise WorkerError('Confirmed baseline changed during worker run')
    return result


def _execute(request):
    """Child-only analysis. Its imports are intentionally limited to read-only code."""
    from .recording import H5Recording
    from . import qc
    request_copy=dict(request);claimed=request_copy.pop('request_sha256',None)
    if digest(request_copy)!=claimed:raise WorkerError('Request digest mismatch')
    config=Config.from_dict(request['config'])
    if digest(config.to_dict())!=request['config_sha256'] or config.binding_sha256()!=request['binding_sha256'] or config.session_id!=request['session_id']:
        raise WorkerError('Request configuration/session binding mismatch')
    kind,_=_stage(request['stage'])
    try:
        config.validate_binding()
        if _stat(request['source']['path'])!=request['source']:raise WorkerError('Source differs from request before opening')
        with H5Recording(request['source']['path'],config) as recording:
            if kind=='preflight':
                result=qc._common(recording,config,'preflight')
                events=recording.events
                if len(events)!=1 or events[0].get('id')!=7001 or events[0].get('eventtype')!=1 or events[0].get('phase')!='preflight' or events[0].get('session')!=config.session_id:
                    result['errors'].append('preflight_requires_unique_7001_type1_marker')
                if recording.duration<10.:
                    result['errors'].append('preflight_less_than_10_seconds')
                result=qc._finish(result)
            elif kind=='recovery':
                result=qc._common(recording,config,'recovery')
                if recording.events:
                    result['errors'].append('recovery_requires_no_events')
                if recording.duration<600.:
                    result['errors'].append('recovery_less_than_600_seconds')
                result=qc._finish(result)
            elif kind=='baseline':
                result=qc.baseline(recording,config)
            elif kind=='pilot':
                baseline_source=request['baseline_source']
                if baseline_source is None or _stat(baseline_source['path'])!=baseline_source:
                    raise WorkerError('Confirmed baseline source missing or changed')
                with H5Recording(baseline_source['path'],config) as baseline_recording:
                    result=qc.pilot(recording,baseline_recording,config,request['parameters']['candidate_roi'])
            else:
                p=request['parameters']
                result=qc.pair(recording,config,p['pair_spec'],p['roi'],p['reference_rate'],p['expected_post'])
            result['qc_stage']=result['stage'];result['stage']=request['stage']
            # Also check unchanged after metadata-only preflight and the final raw read.
            if _stat(request['source']['path'])!=request['source']:raise WorkerError('Source changed during analysis')
        return _envelope(request,result)
    except Exception as error:
        return _envelope(request,_failed(request,type(error).__name__+': '+str(error)))


class ClosedFileAnalyzer:
    def __init__(self,config,run_dir,clock,guard):
        self.config=config;self.run_dir=Path(run_dir).resolve();self.clock=clock;self.guard=guard
        self.qc_dir=self.run_dir/'qc'

    def _baseline(self):
        path=self.qc_dir/'baseline.json'
        if not path.exists():raise WorkerError('No confirmed baseline receipt available for pilot')
        receipt=_load_json(path);result=receipt.get('result',{})
        if receipt.get('config_sha256')!=digest(self.config.to_dict()) or receipt.get('session_id')!=self.config.session_id or receipt.get('binding_sha256')!=self.config.binding_sha256():
            raise WorkerError('Confirmed baseline configuration/session mismatch')
        if result.get('status')!='pass' or receipt.get('result_sha256')!=digest(result):raise WorkerError('Baseline receipt not qualified or digest changed')
        source=receipt.get('source')
        if not isinstance(source,dict) or source!=result.get('source') or _stat(source['path'])!=source:
            raise WorkerError('Confirmed baseline file changed')
        return source,result['candidate_roi']

    def __call__(self,stage,path,**kwargs):
        kind,limit=_stage(stage)
        self.config.validate_binding()
        self.guard()
        source=_stat(path);baseline_source=None
        if kind=='pilot':
            baseline_source,candidate=self._baseline()
            if set(kwargs)-{'candidate_roi','baseline_path'}:raise WorkerError('Unknown pilot QC parameters')
            if 'baseline_path' in kwargs and Path(kwargs['baseline_path']).resolve()!=Path(baseline_source['path']):
                raise WorkerError('Pilot baseline path differs from confirmed receipt')
            if 'candidate_roi' in kwargs and kwargs['candidate_roi']!=candidate:raise WorkerError('Pilot candidates differ from confirmed baseline')
            parameters={'candidate_roi':candidate}
        elif kind=='pair':
            required={'pair_spec','roi','reference_rate','expected_post'}
            if set(kwargs)!=required:raise WorkerError('Pair QC requires pair_spec, roi, reference_rate, expected_post')
            if kwargs['pair_spec'].get('pair')!=int(stage.split('_')[1]):raise WorkerError('Pair stage/spec mismatch')
            parameters=kwargs
        else:
            if kwargs:raise WorkerError('Unexpected QC stage parameters')
            parameters={}
        nonce=uuid.uuid4().hex
        request={'schema_version':1,'request_id':nonce,'stage':stage,'config':self.config.to_dict(),
                 'config_sha256':digest(self.config.to_dict()),'binding_sha256':self.config.binding_sha256(),
                 'session_id':self.config.session_id,'source':source,'baseline_source':baseline_source,'parameters':parameters}
        request['request_sha256']=digest(request)
        request_path=self.qc_dir/(stage+'.'+nonce+'.request.json')
        response_path=self.qc_dir/(stage+'.'+nonce+'.response.json')
        log_path=self.qc_dir/(stage+'.'+nonce+'.worker.log')
        receipt_path=self.qc_dir/(stage+'.json')
        _write_json(request_path,request)
        process=None;start_real=time.monotonic();start_clock=self.clock.mono();last_clock=start_clock
        if not math.isfinite(start_clock):raise WorkerError('Invalid analysis clock')

        def check():
            nonlocal last_clock
            self.guard()
            now=self.clock.mono()
            if not math.isfinite(now) or now<last_clock:raise WorkerError('QC monotonic clock invalid or rolled back')
            last_clock=now
            if max(time.monotonic()-start_real,now-start_clock)>=limit:
                raise WorkerError('%s QC timeout after %.0f seconds'%(stage,limit))

        # Explicit environment allowlist does not forward SDK endpoint/token variables.
        permitted={'SYSTEMROOT','WINDIR','PATH','PATHEXT','TEMP','TMP','TMPDIR','HOME','USERPROFILE',
                   'LANG','LC_ALL','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','HDF5_PLUGIN_PATH','PYTHONPATH'}
        environment={k:v for k,v in os.environ.items() if k.upper() in permitted}
        package_root=str(Path(__file__).resolve().parent.parent)
        environment['PYTHONPATH']=package_root+os.pathsep+environment.get('PYTHONPATH','')
        environment['PYTHONNOUSERSITE']='1';environment['PYTHONDONTWRITEBYTECODE']='1'
        command=[sys.executable,'-m','mea_stim_v2.worker','--request',str(request_path),'--response',str(response_path)]
        options={'cwd':package_root,'env':environment,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
        else:options['start_new_session']=True
        try:
            with log_path.open('xb') as logfile:
                process=subprocess.Popen(command,stdout=logfile,stderr=logfile,**options)
                while True:
                    check()
                    if process.poll() is not None:break
                    time.sleep(.05)
                if process.returncode!=0:raise WorkerError('QC worker exited with code '+str(process.returncode)+'; log: '+str(log_path))
            if not response_path.exists():raise WorkerError('QC worker exited without a result')
            response=_load_json(response_path)
            result=_validate_response(request,response)
            check()
            response['request_path']=str(request_path)
            response['response_path']=str(response_path)
            response['worker_log']=str(log_path)
            response['elapsed_s']=time.monotonic()-start_real
            _write_json(receipt_path,response)
            return dict(result,request_sha256=request['request_sha256'],receipt_path=str(receipt_path))
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.kill()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    raise WorkerError('Owned QC worker could not be reaped after kill') from error
            failed=_envelope(request,_failed(request,type(error).__name__+': '+str(error)))
            failed['worker_pid']=process.pid if process is not None else None
            failed['worker_log']=str(log_path);failed['elapsed_s']=time.monotonic()-start_real
            _write_json(receipt_path,failed)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='Read-only MaxOne closed-file QC worker')
    parser.add_argument('--request',required=True);parser.add_argument('--response',required=True)
    args=parser.parse_args(argv)
    request=_load_json(args.request)
    response=_execute(request)
    _write_json(args.response,response)
    return 0


if __name__=='__main__':raise SystemExit(main())
