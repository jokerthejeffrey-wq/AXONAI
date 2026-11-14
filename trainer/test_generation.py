import unittest

class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except ImportError: raise unittest.SkipTest('PyTorch required; also tested in Windows build')
        cls.torch=torch
        torch.set_num_threads(2)

    def test_cached_logits_match_full_forward(self):
        from student import Student
        t=self.torch
        model=Student(dict(vocab_size=260,width=32,layers=2,heads=4,context=32)).eval()
        ids=t.tensor([[257,104,105,259,258,72]])
        with t.inference_mode():
            _,cache=model.incremental(ids[:,:3])
            for end in range(4,7):
                actual,cache=model.incremental(ids[:,end-1:end],cache)
                expected=model(ids[:,:end])[:,-1:]
                t.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)

    def test_stream_and_budget(self):
        from student import Student,generate_stream
        model=Student(dict(vocab_size=260,width=32,layers=2,heads=4,context=32))
        chunks=list(generate_stream(model,[dict(role='user',content='hello')],max_tokens=10))
        self.assertTrue(all(isinstance(x,str) for x in chunks))
        self.assertLessEqual(len(chunks),11)
        with self.assertRaises(ValueError):
            list(generate_stream(model,[],temperature=0))
