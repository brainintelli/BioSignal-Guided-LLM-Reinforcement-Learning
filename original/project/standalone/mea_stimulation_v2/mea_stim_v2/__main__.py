"""Explicit live entry point; all other package commands are read-only."""
import argparse,json,signal,sys
from pathlib import Path
from .config import Config
from .protocol import RealClock

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 t=s.add_parser('template');t.add_argument('path')
 v=s.add_parser('validate');v.add_argument('config')
 l=s.add_parser('live');l.add_argument('config');l.add_argument('--output',required=True);l.add_argument('--registry',required=True);l.add_argument('--continue-zero-preflight')
 a=p.parse_args()
 if a.command=='template':
  with Path(a.path).open('x',encoding='utf-8') as f:json.dump(Config.template().to_dict(),f,indent=2)
  return 0
 c=Config.from_dict(json.loads(Path(a.config).read_text(encoding='utf-8')))
 c.validate_binding()
 if a.command=='validate':
  print(json.dumps({'binding_sha256':c.binding_sha256(),'config_valid':True,'hardware_contacted':False}));return 0
 c.validate_live()
 if Path(a.registry).resolve()!=Path('/home/researcher/.local/share/mea_stim_v2/launch_registry'):
  raise ValueError('Live execution requires the fixed acquisition-host cumulative registry')
 from .live import MaxLabBackend
 from .worker import ClosedFileAnalyzer
 from .controller import Controller
 clock=RealClock();backend=MaxLabBackend(c,a.output,clock)
 controller=Controller(c,a.output,a.registry,backend,None,clock,resume_zero_preflight=a.continue_zero_preflight)
 controller.analyze=ClosedFileAnalyzer(c,a.output,clock,controller.guard)
 def interrupted(signum,frame):raise KeyboardInterrupt('Deadline reserve/signal '+str(signum))
 for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(signum,interrupted)
 def defer_signals_for_cleanup():
  for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(signum,signal.SIG_IGN)
  signal.setitimer(signal.ITIMER_REAL,0)
 controller.before_cleanup=defer_signals_for_cleanup
 remaining=c.started_wall()+7200-clock.wall()-120
 if remaining<=0:raise ValueError('Cleanup deadline already reached')
 signal.setitimer(signal.ITIMER_REAL,remaining)
 try:result=controller.run()
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 print(json.dumps({'status':result['status'],'dose_totals':result['dose_totals'],'errors':result['errors'],'cleanup_errors':result['cleanup_errors']}),flush=True)
 return 0 if result['status']=='completed' else 2

if __name__=='__main__':sys.exit(main())
