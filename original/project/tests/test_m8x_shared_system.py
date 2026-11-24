"""A prefix may outlive calls only without gradients and at the same weights."""
import importlib
import unittest
import importlib.util
torch=importlib.import_module('torch') if importlib.util.find_spec('torch') else None


@unittest.skipUnless(torch is not None and importlib.util.find_spec('transformers') and importlib.util.find_spec('triton'),'Node-5 runtime required')
class PrefixBankTests(unittest.TestCase):
    def test_inference_cache_invalidates_and_separates_reference(self):
        bank=importlib.import_module('agent.m8x_shared_system').PrefixBank()
        calls=[]
        def build():
            calls.append(1)
            return object()
        with torch.no_grad():
            first=bank.acquire(False,('v1',),{},build)
            self.assertIs(first,bank.acquire(False,('v1',),{},build))
            self.assertIsNot(first,bank.acquire(True,('v1',),{},build))
            self.assertIsNot(first,bank.acquire(False,('v2',),{},build))
        self.assertEqual(len(calls),3)

    def test_training_prefix_is_graph_local_and_never_detached(self):
        bank=importlib.import_module('agent.m8x_shared_system').PrefixBank()
        weight=torch.tensor(3.,requires_grad=True)
        calls=[]
        def build():
            calls.append(1)
            return weight.square()
        with torch.no_grad():bank.acquire(False,('v1',),{},build)
        scope={}
        a=bank.acquire(False,('v1',),scope,build)
        b=bank.acquire(False,('v1',),scope,build)
        self.assertIs(a,b)
        (a+b).backward()
        self.assertEqual(weight.grad.item(),12.)
        c=bank.acquire(False,('v1',),{},build)
        self.assertIsNot(a,c)
        c.backward()
        self.assertEqual(weight.grad.item(),18.)
        self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main()
