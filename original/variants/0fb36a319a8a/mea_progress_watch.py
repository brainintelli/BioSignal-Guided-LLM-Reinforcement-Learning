"""Consume the growing sequenced file on the same host, before acquisition ends."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.mea_timing import FilteredFrameStream


def watch(events_path,output,timeout):
    gate=FilteredFrameStream();source=None;count=0;max_age=0.;error=None
    deadline=time.monotonic()+timeout;started=time.time_ns()
    with output.open('x') as out:
        try:
            while time.monotonic()<deadline and not gate.ended:
                if source is None:
                    if not events_path.exists():time.sleep(.01);continue
                    source=events_path.open('rb')
                data=source.read(65536)
                if not data:time.sleep(.01);continue
                rows=gate.feed(data)
                now=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                for row in rows:
                    if row['type']=='frame_progress':
                        age=(now-row['recv_end_ns'])/1e6
                        if age<0:raise ValueError('watcher and producer clocks differ')
                        max_age=max(max_age,age);count+=1
                        out.write(json.dumps(dict(type='observed_progress',source_seq=row['seq'],
                            covered_end_frame=row['end_frame'],observed_mono_raw_ns=now,
                            age_ms=age,eligible=age<=250))+'\n');out.flush()
                    elif row['type']=='frame_heartbeat':max_age=max(max_age,row['since_frame_ns']/1e6)
            if not gate.ended:raise TimeoutError('live watcher deadline')
            gate.finish()
        except Exception as exc:error=str(exc)
        finally:
            if source:source.close()
        result=dict(type='watch_end',passed=error is None and max_age<=250,errors=[] if error is None else [error],
                    started_unix_ns=started,ended_unix_ns=time.time_ns(),progress_updates=count,
                    maximum_observed_age_ms=max_age,final_watermark=gate.watermark,event_count=gate.event_count)
        out.write(json.dumps(result)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=90)
    args=p.parse_args()
    if not 1<=args.timeout<=120:p.error('timeout must be 1..120 seconds')
    def stop(signum,frame):raise InterruptedError('watcher stopped')
    signal.signal(signal.SIGTERM,stop)
    result=watch(args.events,args.output,args.timeout);print(json.dumps(result))
    return 0 if result['passed'] else 2


if __name__=='__main__':sys.exit(main())
