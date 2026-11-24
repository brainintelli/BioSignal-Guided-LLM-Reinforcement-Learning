"""Hard parent deadline for model initialization, GPU collectives and updates."""
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import write

def group_active(pgid):
    """Linux live supervisor: zombies hold no device/compute resources."""
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:fields=(item/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        if int(fields[2])==pgid and fields[0] not in ('Z','X'):return True
    return False

def stop_tree(process,grace):
    if os.name=='posix':
        # The leader can exit before its workers. Its PID still identifies the
        # private group created by start_new_session; always clean that group.
        def send(sig):
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
        send(signal.SIGTERM);end=time.monotonic()+grace
        while group_active(process.pid) and time.monotonic()<end:
            process.poll();time.sleep(.01)
        if group_active(process.pid):send(signal.SIGKILL)
        process.wait(timeout=5);end=time.monotonic()+5
        while group_active(process.pid) and time.monotonic()<end:time.sleep(.01)
        if group_active(process.pid):raise RuntimeError('Owned process group failed to stop')
    elif process.poll() is None:
        process.terminate()
        try:process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
            process.kill();process.wait(timeout=5)

def run(command,output,budget,grace=5.):
    output=Path(output);began=time.monotonic();process=None
    result=dict(status='starting',returncode=None,hard_deadline_wall=budget.wall_end,real_stimulation=False,r2_enabled=False)
    try:
        budget.require()
        with (output/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=os.name=='posix')
            result['pid']=process.pid
            while process.poll() is None:
                try:budget.require()
                except TimeoutError:
                    result['status']='deadline_stopped';stop_tree(process,grace);break
                time.sleep(.05)
            if result['status']=='starting':result['status']='completed' if process.returncode==0 else 'worker_failed'
    except BaseException as e:
        result.update(status='failed',error=type(e).__name__+': '+str(e))
    finally:
        if process is not None:
            stop_tree(process,grace);result['returncode']=process.poll()
        result['elapsed_seconds']=time.monotonic()-began
        write(output/'supervisor.json',result)
    return result
