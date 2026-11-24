"""Regression for pinned saved-tensor cache retention at a quiescent boundary."""
import gc,unittest
from pathlib import Path
import torch
try:from host_cache import release_host_cache
except ImportError:
    def release_host_cache():pass
def rss():return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
class HostCacheTests(unittest.TestCase):
    def test_released_pinned_buffers_stop_retaining_rss(self):
        torch.set_num_threads(2);torch.cuda.synchronize();torch._C._host_emptyCache();gc.collect();baseline=rss()
        blocks=[torch.empty(n*1024**2,dtype=torch.uint8,pin_memory=True).zero_() for n in [64,128]]
        del blocks;gc.collect();cached=rss();self.assertGreater(cached-baseline,128*1024**2)
        release_host_cache();remaining=rss()-baseline
        print('pinned_cache_bytes_before',cached-baseline,'after',remaining,flush=True)
        self.assertLess(remaining,64*1024**2)
    def test_live_pinned_buffer_and_copy_survive_cleanup(self):
        x=torch.arange(1024,dtype=torch.float32,pin_memory=True);y=x.to('cuda',non_blocking=True)
        release_host_cache();self.assertTrue(torch.equal(x,y.cpu()));self.assertTrue(x.is_pinned())
if __name__=='__main__':unittest.main()
