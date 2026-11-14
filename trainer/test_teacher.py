import io
import unittest
from contextlib import redirect_stdout
from trainer.progress import bar, duration
from trainer.teacher import Teacher

class LocalTeacherTests(unittest.TestCase):
    def sample(self):
        return {'messages': [
            {'role': 'user', 'content': 'What is the capital city of France?'},
            {'role': 'assistant', 'content': 'Paris.'}
        ]}

    def test_helper_starts_without_pretrained_dependencies(self):
        with redirect_stdout(io.StringIO()) as capture:
            teacher = Teacher('cpu')
        self.assertIn('zero pretrained downloads', capture.getvalue())
        self.assertFalse(hasattr(teacher, 'model'))
        teacher.close()

    def test_source_structure_checks(self):
        teacher=Teacher()
        self.assertTrue(teacher.review_source(self.sample()))
        self.assertFalse(teacher.review_source({'messages': [
            {'role':'user','content':'Question?'},
            {'role':'assistant','content':'Question?'}]}))
        self.assertFalse(teacher.review_source({'messages': [
            {'role':'user','content':'Question?'},
            {'role':'assistant','content':'Answer\x00'}]}))
        self.assertFalse(teacher.review_source({'messages': []}))

    def test_long_source_answer_is_still_eligible_for_short_context_training(self):
        teacher=Teacher()
        long_source={'messages':[{'role':'user','content':'Summarize this topic.'},
                                 {'role':'assistant','content':('Useful explanatory text. '*50)}]}
        self.assertTrue(teacher.review_source(long_source))
        # The teacher must still reject unsafe control characters.
        long_source['messages'][1]['content']+='\\x00'
        self.assertFalse(teacher.review_source(long_source))

    def test_variation_preserves_original_answer(self):
        teacher=Teacher()
        source=self.sample()
        generated=teacher.generate_checked(source,24)
        self.assertIsNotNone(generated)
        self.assertEqual(generated['response'],'Paris.')
        self.assertNotEqual(generated['messages'][-1]['content'],source['messages'][0]['content'])
        self.assertIn(source['messages'][0]['content'],generated['messages'][-1]['content'])
        self.assertEqual(source['messages'][0]['content'],'What is the capital city of France?')

    def test_sampling_and_context_budget(self):
        teacher=Teacher()
        self.assertIsNone(teacher.generate_checked(self.sample(),25))
        too_long={'messages':[{'role':'user','content':'Q'*400},{'role':'assistant','content':'A'}]}
        self.assertIsNone(teacher.generate_checked(too_long,24))

    def test_ascii_bar_and_eta(self):
        display=bar(25,'Installing',eta=75)
        self.assertIn('[=====',display)
        self.assertIn('25%',display)
        self.assertIn('~1m 15s',display)
        self.assertEqual(duration(2),'~2s')

if __name__=='__main__':
    unittest.main()
