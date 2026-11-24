"""Grouped GEMMs must retain all independent adapter gradients."""
import importlib
import unittest
import importlib.util
torch=importlib.import_module('torch') if importlib.util.find_spec('torch') else None


@unittest.skipUnless(torch is not None and importlib.util.find_spec('transformers') and importlib.util.find_spec('triton'),'Node-5 runtime required')
class GroupedProjectionTests(unittest.TestCase):
    def test_execution_cache_refreshes_once_per_weight_version_and_reference(self):
        cache=importlib.import_module('agent.m8x_grouped_lora').ExecutionCache()
        calls=[]
        for reference,version in [(False,1),(False,1),(True,1),(False,2),(False,2)]:
            cache.prepare(reference,version,lambda:calls.append(1))
        self.assertEqual(len(calls),3)
        cache.revisions.clear()
        cache.prepare(False,2,lambda:calls.append(1))
        self.assertEqual(len(calls),4)

    def fixture(self):
        torch.manual_seed(101)
        rows=[]
        for out,rank,scale in [(7,2,.5),(5,3,2.),(9,2,1.)]:
            rows.append((torch.randn(out,11,dtype=torch.float64),
                         torch.randn(rank,11,dtype=torch.float64,requires_grad=True),
                         torch.randn(out,rank,dtype=torch.float64,requires_grad=True),scale))
        return rows

    def test_output_input_and_every_adapter_gradient_match_independent_dense_reference(self):
        cls=importlib.import_module('agent.m8x_grouped_lora').GroupedProjection
        rows=self.fixture();reference=[tuple(t.detach().clone().requires_grad_(t.requires_grad) if torch.is_tensor(t) else t for t in row) for row in rows]
        x=torch.randn(2,4,11,dtype=torch.float64,requires_grad=True);xx=x.detach().clone().requires_grad_()
        got=cls(rows)(x)
        expected=[torch.nn.functional.linear(xx,w+s*(b@a)) for w,a,b,s in reference]
        for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        upstream=[torch.randn_like(a) for a in got]
        sum((a*g).sum() for a,g in zip(got,upstream)).backward()
        sum((a*g).sum() for a,g in zip(expected,upstream)).backward()
        torch.testing.assert_close(x.grad,xx.grad,rtol=1e-11,atol=1e-11)
        for row,ref in zip(rows,reference):
            for ix in (1,2):torch.testing.assert_close(row[ix].grad,ref[ix].grad,rtol=1e-11,atol=1e-11)

    def test_optimizer_refreshes_cache_without_changing_base(self):
        cls=importlib.import_module('agent.m8x_grouped_lora').GroupedProjection
        rows=self.fixture();base=[r[0].clone() for r in rows];group=cls(rows)
        x=torch.randn(5,11,dtype=torch.float64);params=[p for r in rows for p in r[1:3]]
        optimizer=torch.optim.AdamW(params,lr=.01)
        before=group(x);sum(y.square().mean() for y in before).backward();optimizer.step()
        after=group(x)
        self.assertGreater(max(float((a-b).abs().max()) for a,b in zip(before,after)),1e-5)
        for y,(w,a,b,s),saved in zip(after,rows,base):
            torch.testing.assert_close(y,torch.nn.functional.linear(x,w+s*(b@a)),rtol=1e-12,atol=1e-12)
            self.assertTrue(torch.equal(w,saved))


if __name__=='__main__':unittest.main()
