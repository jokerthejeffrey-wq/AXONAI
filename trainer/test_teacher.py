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

    def test_double_review_required_before_training(self):
        teacher,history=self.fake()
        row=teacher.generate_checked(self.sample(),24)
        self.assertIsNotNone(row)
        self.assertEqual(len(history),3)
        self.assertEqual(row['response'],'Paris.')

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
