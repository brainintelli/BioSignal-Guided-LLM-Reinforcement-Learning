import unittest
from process_identity import is_worker
class ProcessIdentityTests(unittest.TestCase):
    def test_launcher_is_not_a_worker(self):
        script=b'/task/eightgpu_run.py';out=b'/task/joint02'
        self.assertTrue(is_worker([b'/runtime/python3.12',b'-u',script,b'collect',out,b''],script,out))
        self.assertFalse(is_worker([b'/runtime/python3.12',b'-m',b'torch.distributed.run',script,b'collect',out],script,out))
        self.assertFalse(is_worker([b'/runtime/python3.12',b'-u',script,b'verify',out],script,out))
        self.assertFalse(is_worker([b'/runtime/python3.12',b'-u',script,b'collect',b'/task/other'],script,out))
        self.assertFalse(is_worker([],script,out))
if __name__=='__main__':unittest.main()
