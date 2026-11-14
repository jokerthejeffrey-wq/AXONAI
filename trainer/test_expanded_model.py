"""Checks that the expanded architecture is real, trainable, and isolated."""
import unittest

from settings import CONFIG, LARGE_CONFIG, parameter_count


class ProfileTests(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(parameter_count(LARGE_CONFIG), 50_224_640)
        self.assertGreater(parameter_count(LARGE_CONFIG), parameter_count(CONFIG)*10)

    def test_research_model_preserves_legacy_profile(self):
        self.assertEqual(CONFIG, dict(vocab_size=260,width=256,layers=4,heads=8,context=512))
        self.assertEqual(LARGE_CONFIG['context'], 1024)


class TorchExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('PyTorch is not installed in the lightweight CI environment')
        cls.torch=torch

    def test_large_transformer_and_transfer(self):
        from student import Student
        from trainer.expanded_model import expand_legacy
        torch=self.torch
        torch.set_num_threads(2)
        small=Student()
        large=expand_legacy(small,seed=50224)
        self.assertEqual(sum(x.numel() for x in large.parameters()), 50_224_640)
        self.assertEqual(large.config,LARGE_CONFIG)
        self.assertTrue(torch.equal(large.token.weight[:,:256],small.token.weight))
        self.assertTrue(torch.equal(large.blocks[0].qkv.weight[:256,:256],
                                    small.blocks[0].qkv.weight[:256,:]))
        self.assertTrue(torch.equal(large.head.weight[:,:256],small.head.weight))
        # A real forward/backward pass must update at least one internal layer.
        before=large.blocks[0].qkv.weight.detach().clone()
        optimizer=torch.optim.AdamW(large.parameters(),lr=1e-4)
        batch=torch.tensor([[257,104,105,259,258,72]],dtype=torch.long)
        loss=torch.nn.functional.cross_entropy(
            large(batch[:,:-1]).reshape(-1,260),batch[:,1:].reshape(-1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(large.blocks[0].qkv.weight,before))


if __name__=='__main__':
    unittest.main()
