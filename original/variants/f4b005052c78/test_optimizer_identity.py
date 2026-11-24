import tempfile,unittest
from pathlib import Path
import torch
from eightgpu_run import optimizer_hash,saved_optimizer_hash

class OptimizerIdentityTests(unittest.TestCase):
    def make(self):
        result={}
        for name in ['actor','critic']:
            p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.01);p.grad=torch.tensor([.1,.2]);opt.step();result[name]=opt
        return result
    def test_scalar_steps_and_disk_roundtrip(self):
        opts=self.make();h=optimizer_hash(opts)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';torch.save({k:v.state_dict() for k,v in opts.items()},path)
            loaded,steps=saved_optimizer_hash(path)
        self.assertEqual(loaded,h);self.assertEqual(steps,{'actor':[1],'critic':[1]})
    def test_changed_optimizer_tensor_is_detected(self):
        opts=self.make();h=optimizer_hash(opts);next(iter(opts['actor'].state.values()))['step'].add_(1)
        self.assertNotEqual(optimizer_hash(opts),h)
if __name__=='__main__':unittest.main()
