"""Acquisition worker, explicitly bound at runtime. An idle SSH channel has no authority."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from mea_binary_r2.encoding import Intent
from mea_state2_r2.features import digest
from .device import Device,write_once
from .protocol import binding,transaction
from .predictability import MODE
from .qualification import VERSION, analysis, order_for, require_pass, trial_row


class Session:
    def __init__(self):
        self.dev=None
        self.started=False

    def protocol(self, feedback_encoding=None):
        if feedback_encoding is None:
            return dict(qualification=VERSION)
        if feedback_encoding != MODE:
            raise ValueError('Unknown feedback encoding')
        return dict(qualification=VERSION, feedback_encoding=MODE)

    def open(self,b):
        if self.dev is not None:raise ValueError('Already open')
        remaining=binding(b)+7200-time.time()
        if not 180<remaining<=7200:raise ValueError('Invalid biological start')
        out=Path(b['record_root']).resolve()/b['session_id']
        if not str(out).isascii() or not out.is_absolute():raise ValueError('ASCII recording path required')
        out.mkdir(parents=True,exist_ok=False)
        self.dev=Device(b,out)
        signal.setitimer(signal.ITIMER_REAL,remaining-130)
        self.dev.read_calibration()
        self.dev.verify_selection()
        self.dev.claim()
        self.dev.prepare_sites()
        windows=self.dev.prime('baseline',True)
        self.dev.verify_output_calibration()
        return dict(binding=self.dev.bid,windows=windows,norm=self.dev.norm,out=str(out))

    def calibrate(self):
        d=self.dev
        if self.started:raise ValueError('Calibration cannot be repeated')
        self.started=True
        order=order_for(d.bid)
        report=dict(schema=VERSION,binding=d.bid,order=order,trials=[],passed=False)
        write_once(d.run_dir/'qualification-plan.json',dict(schema=VERSION,binding=d.bid,order=order))
        try:
            if d.attempts+9>30:raise ValueError('Insufficient remaining attempts for nine complete trials')
            quiet=transaction(d.bid,'technical','qualification',0,-1,digest(['quiet']),Intent(None,0,True,()))
            report['quiet_receipt']=d.slot(quiet,calibration=True)['id']
            for block,sites in enumerate(order):
                for j,site in enumerate(sites):
                    req=transaction(d.bid,'technical','qualification',block,block*3+j,
                                    digest(['qualification',block,site]),Intent(site,3,True,(site,)))
                    receipt=d.slot(req,calibration=True)
                    report['trials'].append(trial_row(receipt,req,d.norm))
            report['analysis']=analysis(d.bid,report['trials'])
            report['passed']=report['analysis']['passed']
        except (RuntimeError,ValueError,TimeoutError,OSError,KeyError) as exc:
            report['error']=str(exc)
        if not report['passed']:
            report['shutdown_errors']=d.cleanup()
        report['attempts']=d.attempts
        write_once(d.run_dir/'qualification.json',report)
        if report['passed']:
            require_pass(report,d.bid)
            d.response_passed=True
        return report

    def prime(self,name):
        if not self.dev.response_passed:raise ValueError('Qualification required before trial priming')
        return self.dev.prime(name)

    def slot(self,req):
        return self.dev.slot(req)


def main():
    wire=os.fdopen(os.dup(1),'w',encoding='utf-8',buffering=1)
    os.dup2(sys.stderr.fileno(),1)
    def expire(*_):
        raise TimeoutError('Acquisition deadline or disconnect')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,expire)
    session=Session()
    try:
        for line in sys.stdin:
            cmd=json.loads(line)
            if cmd['op']=='close':break
            if cmd['op'] not in ('protocol','open','calibrate','prime','slot'):raise ValueError('Unknown acquisition operation')
            result=getattr(session,cmd['op'])(**cmd.get('args',{}))
            print(json.dumps(dict(ok=True,value=result),allow_nan=False),file=wire,flush=True)
    except BaseException:
        print(json.dumps(dict(ok=False,error=traceback.format_exc())),file=wire,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if session.dev:
            d=session.dev
            errors=d.cleanup()
            write_once(d.run_dir/'closed.json',dict(errors=errors,attempts=d.attempts,time=time.time()))
            if hasattr(d,'lease'):d.lease.close()
            print(json.dumps(dict(ok=not errors,value=dict(closed=True,errors=errors))),file=wire,flush=True)


if __name__=='__main__':main()
