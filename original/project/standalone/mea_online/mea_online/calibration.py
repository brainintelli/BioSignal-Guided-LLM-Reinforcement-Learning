"""Training-before-only normalization and independent engineering holdout.

This checks normalization usability, not tissue health or stimulation recovery.
"""
from pathlib import Path
from statistics import median,pstdev
import time
import uuid
from .adapter import validate_window
from .common import append,digest,write
from .trainer import features

def collect(source,binding,output,budget,*,live=True,clock=time):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    values=[];ids=[];last=None;nonce=uuid.uuid4().hex
    try:
        for i in range(90):
            budget.require(9);request_id='cal-'+nonce+'-'+str(i)
            began=clock.monotonic();raw=source.capture(request_id)
            if not 0<=clock.monotonic()-began<=8:raise TimeoutError('Calibration window response timed out')
            budget.require();w=validate_window(raw,binding,request_id,last,live=live)
            if w['window_sha256'] in ids:raise ValueError('Repeated calibration evidence')
            append(out/'windows.jsonl',w);last=w['end_frame'];ids.append(w['window_sha256']);values.append(features(w,binding))
        # Floors handle discrete/count features; they are not biological targets.
        floors=[1.,.01,.01,1.]
        centers=[median([v[j] for v in values[:60]]) for j in range(4)]
        scales=[max(floors[j],1.4826*median([abs(v[j]-centers[j]) for v in values[:60]]),pstdev([v[j] for v in values[:60]])) for j in range(4)]
        flags=[any(abs((v[j]-centers[j])/scales[j])>5 for j in range(4)) for v in values[60:]]
        report=dict(schema='mea.online.normalization-validation.v1',calibration_windows=60,validation_windows=30,
          outlier_windows=sum(flags),allowed_outlier_windows=6,scale_floors=floors,scale_rule='max_floor_scaledMAD_populationSD',biological_qc_passed=False,
          interpretation='Normalization holdout only; transport counts are not spike-peak response windows',source='live' if live else 'simulation')
        write(out/'validation.json',report)
        if sum(flags)>6:raise ValueError('Independent normalization validation failed')
        c=dict(schema='mea.online.calibration.v1',binding_sha256=digest(binding),source='live' if live else 'simulation',
          coverage_basis='filtered_transport_frames',centers=centers,scales=scales,
          calibration_window_ids=ids[:60],validation_window_ids=ids[60:],status='engineering_validated')
        c['normalization_id']=digest(c);write(out/'calibration.json',c)
        return c
    except BaseException as e:
        write(out/'failure.json',dict(error=type(e).__name__+': '+str(e),received_windows=len(ids),r2_enabled=False))
        raise
