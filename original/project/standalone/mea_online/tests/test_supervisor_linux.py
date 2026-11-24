import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from mea_online.supervisor import stop_tree

def active(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False

@unittest.skipUnless(sys.platform=='linux','Linux process-group termination test')
class ProcessGroupTests(unittest.TestCase):
    def case(self,leader_exits):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'child.pid'
            child="import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(30)"%str(marker)
            parent="import subprocess,sys,time,pathlib,os;subprocess.Popen([sys.executable,'-c',%r]);p=pathlib.Path(%r)\nwhile not p.exists():time.sleep(.01)\n%s"%(child,str(marker),'os._exit(0)' if leader_exits else 'time.sleep(30)')
            proc=subprocess.Popen([sys.executable,'-c',parent],start_new_session=True)
            try:
                end=time.monotonic()+5
                while not marker.exists() and time.monotonic()<end:time.sleep(.01)
                self.assertTrue(marker.exists());pid=int(marker.read_text())
                if leader_exits:proc.wait(timeout=5)
                stop_tree(proc,.05)
                self.assertFalse(active(pid),'Surviving child after leader exit')
                self.assertIsNotNone(proc.poll())
            finally:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait(timeout=5)

    def test_child_ignoring_term_cannot_outlive_terminated_parent(self):self.case(False)
    def test_already_exited_parent_still_requires_group_cleanup(self):self.case(True)

if __name__=='__main__':unittest.main()
