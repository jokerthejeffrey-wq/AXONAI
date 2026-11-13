import io
import json
import unittest
from trainer.progress import bar,duration
from trainer.teacher import Teacher,_candidate_from_json,approved


class ReviewerTests(unittest.TestCase):
    def sample(self):
        return {'messages':[
            {'role':'user','content':'What is the capital city of France?'},
            {'role':'assistant','content':'Paris.'}
        ]}

    def fake(self,decisions=('PASS','PASS')):
        teacher=object.__new__(Teacher)
        history=[]
        output=json.dumps({'question':'Which city is the capital of France?',
                           'answer':'Paris.'})
        results=iter([output,*decisions])
        def respond(system,user,max_new_tokens=120):
            history.append((system,user))
            return next(results)
        teacher._ask=respond
        return teacher,history

    def test_local_checker_loader_reuses_snapshot_and_modern_dtype(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        models,tokenizers,downloads=[],[],[]
        class FakeModel:
            def to(self, device):
                self.device=device
                return self
            def eval(self):return self
        class Model:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                models.append((args,kwargs))
                return FakeModel()
        class Tokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                tokenizers.append((args,kwargs))
                return object()
        mock_torch=ModuleType('torch')
        mock_torch.float16='float16'
        mock_hub=ModuleType('huggingface_hub')
        def snapshot_download(**kwargs):
            downloads.append(kwargs)
            return '/fake/cached/checker'
        mock_hub.snapshot_download=snapshot_download
        mock_transformers=ModuleType('transformers')
        mock_transformers.AutoTokenizer=Tokenizer
        mock_transformers.AutoModelForCausalLM=Model
        with patch.dict(sys.modules, {'torch':mock_torch,
                                      'huggingface_hub':mock_hub,
                                      'transformers':mock_transformers}):
            teacher=Teacher('cpu')
        self.assertIsNotNone(teacher.model)
        self.assertEqual(len(downloads),1)
        self.assertEqual(downloads[0]['max_workers'],2)
        self.assertEqual(len(models),1)
        self.assertEqual(models[0][1]['dtype'],'float16')
        self.assertNotIn('torch_dtype',models[0][1])
        self.assertEqual(models[0][1]['local_files_only'],True)
        self.assertEqual(tokenizers[0][1]['local_files_only'],True)

    def test_checker_download_failure_is_actionable(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        mock_torch=ModuleType('torch')
        mock_torch.float16='float16'
        mock_hub=ModuleType('huggingface_hub')
        def fail(**kwargs):
            raise OSError('simulated network problem')
        mock_hub.snapshot_download=fail
        mock_transformers=ModuleType('transformers')
        mock_transformers.AutoTokenizer=object
        mock_transformers.AutoModelForCausalLM=object
        with patch.dict(sys.modules, {'torch':mock_torch,
                                      'huggingface_hub':mock_hub,
                                      'transformers':mock_transformers}):
            with self.assertRaisesRegex(RuntimeError,'Checker download failed'):
                Teacher('cpu')

    def test_double_review_required_before_training(self):
        teacher,history=self.fake()
        row=teacher.generate_checked(self.sample(),24)
        self.assertIsNotNone(row)
        self.assertEqual(len(history),3)
        self.assertEqual(row['response'],'Paris.')

    def test_original_source_also_needs_two_passes(self):
        for decisions,expected,calls in [
            (['PASS','PASS'],True,2),
            (['FAIL','PASS'],False,1),
            (['PASS','FAIL'],False,2),
        ]:
            with self.subTest(decisions=decisions):
                teacher=object.__new__(Teacher)
                results=iter(decisions)
                invocations=[]
                def respond(system,user,max_new_tokens=120):
                    invocations.append((system,user))
                    return next(results)
                teacher._ask=respond
                self.assertEqual(teacher.review_source(self.sample()),expected)
                self.assertEqual(len(invocations),calls)

    def test_each_reviewer_can_reject(self):
        for decisions,count in [(('FAIL','PASS'),2),(('PASS','FAIL'),3)]:
            with self.subTest(decisions=decisions):
                teacher,history=self.fake(decisions)
                self.assertIsNone(teacher.generate_checked(self.sample(),24))
                self.assertEqual(len(history),count)

    def test_unsafe_or_unparseable_answers_rejected(self):
        self.assertIsNone(_candidate_from_json('Not JSON'))
        self.assertIsNone(_candidate_from_json(json.dumps({'question':'Q','answer':'A'})))
        self.assertFalse(approved('PASS maybe'))
        self.assertFalse(approved('FAIL'))

    def test_ascii_bar_and_eta(self):
        display=bar(25,'Installing',eta=75)
        self.assertIn('[=====',display)
        self.assertIn('25%',display)
        self.assertIn('~1m 15s',display)
        self.assertEqual(duration(2),'~2s')


if __name__=='__main__':
    unittest.main()
