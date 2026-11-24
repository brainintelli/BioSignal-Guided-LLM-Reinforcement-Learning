import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools import mea_clock_probe as p


@unittest.skipUnless(os.name=='posix','POSIX SSH stdio transport')
class ClockProbeTests(unittest.TestCase):
    def test_partial_line_is_bounded_by_deadline(self):
        r,w=os.pipe()
        try:
            os.write(w,b'{')
            with os.fdopen(r,'rb',buffering=0) as f:
                with self.assertRaises(TimeoutError):p.read_line(f,.03)
        finally:os.close(w)

    def test_real_local_stdio_exchange_contains_zero_offset(self):
        real_popen=subprocess.Popen
        def local_server(command,**kwargs):return real_popen([sys.executable,str(Path(p.__file__)),'--server'],**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(count=3,interval=.1,output=Path(tmp)/'probe',bind_address=None,
                                    host='unused',remote_script='unused')
            with patch.object(p.subprocess,'Popen',side_effect=local_server):self.assertEqual(p.client(args),0)
            report=json.loads((args.output/'clock_audit.json').read_text())
            lo,hi=report['offset_bounds_ns'];self.assertLessEqual(lo,0);self.assertGreaterEqual(hi,0)
