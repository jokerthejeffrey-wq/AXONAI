import unittest
from trainer.lessons import source_lessons,source_pairs

def row(question,answer):
    return {'messages':[{'role':'user','content':question},
                        {'role':'assistant','content':answer}]}

class LessonExtractionTests(unittest.TestCase):
    def test_exact_short_pair(self):
        output=source_lessons(row('hi','Hello!'))
        self.assertEqual(output[0]['response'],'Hello!')
        self.assertEqual(output[0]['messages'][0]['content'],'hi')
        self.assertEqual(len(output),1)

    def test_long_context_does_not_veto_a_prior_short_turn(self):
        source={'messages':[{'role':'user','content':'Hi'},
                            {'role':'assistant','content':'Hello!'},
                            {'role':'user','content':'x'*1300},
                            {'role':'assistant','content':'Short reply'}]}
        output=source_lessons(source)
        self.assertTrue(any(item['response']=='Hello!' for item in output))
        self.assertFalse(any(item['response']=='Short reply' for item in output))

    def test_long_answer_is_used_as_a_real_source_continuation(self):
        source=row('q'*700,'A helpful explanation. '*45)
        output=source_lessons(source)
        self.assertGreaterEqual(len(output),1)
        self.assertEqual(output[0]['concept'],'source-text-continuation')
        initial=output[0]['messages'][0]['content'].split('\n',1)[1]
        self.assertIn(initial,source['messages'][1]['content'])
        self.assertIn(output[0]['response'],source['messages'][1]['content'])
        self.assertLessEqual(sum(len(x['content'].encode())+2 for x in output[0]['messages'])+
                             len(output[0]['response'].encode())+2,512)

    def test_retains_short_answer_for_long_source_reply(self):
        output=source_lessons(row('Tell me about Paris','Paris is in France. '*100))
        self.assertTrue(output)
        self.assertEqual(output[0]['concept'],'conversation')
        self.assertLessEqual(len(output[0]['response'].encode()),509-2-len('Tell me about Paris'))

    def test_control_characters_are_rejected(self):
        self.assertEqual(source_lessons(row('Hello','Unsafe \x00 response')),[])
        self.assertEqual(source_pairs(row('Hello\x00','safe answer')),[])
        self.assertEqual(source_lessons({'messages':[]}),[])

    def test_multiturn_contains_more_than_one_unique_pair(self):
        source={'messages':[{'role':'user','content':'Hi'},
                            {'role':'assistant','content':'Hello'},
                            {'role':'user','content':'Bye'},
                            {'role':'assistant','content':'Goodbye'}]}
        output=source_lessons(source)
        self.assertEqual({v['response'] for v in output},{'Hello','Goodbye'})
        self.assertLessEqual(len(output),3)

    def test_strict_input_and_bound(self):
        with self.assertRaises(ValueError): source_lessons({},max_lessons=0)
        self.assertEqual(source_lessons(None),[])

if __name__=='__main__':unittest.main()
