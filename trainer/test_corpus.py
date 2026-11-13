import tempfile,unittest,json
from pathlib import Path
from trainer.corpus import prepare,read_rows,training_context
class CorpusTests(unittest.TestCase):
 def test_dedup_cache_and_random_access(self):
  source=[{'messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':v}]} for v in ['Hi','Hello again']]
  source.append({'messages':[{'role':'user','content':'Explain rain'},{'role':'assistant','content':'Water condenses.'}]})
  with tempfile.TemporaryDirectory() as d:
   info=prepare(d,source,10000);self.assertEqual(info['records'],2)
   self.assertEqual(read_rows(d,1,1)[0]['messages'][0]['content'],'Explain rain')
   self.assertEqual(prepare(d,[],10000),info)
   self.assertEqual(len((Path(d)/'training.txt').read_text().splitlines()),2)
 def test_no_duplicate_padding_to_target(self):
  with tempfile.TemporaryDirectory() as d:
   info=prepare(d,[{'messages':[{'role':'user','content':'hi'},{'role':'assistant','content':'hello'}]}],1000000)
   self.assertLess(info['bytes'],1000000);self.assertEqual(info['lines'],1)
 def test_multiturn_context_is_preserved(self):
  row={'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'},{'role':'user','content':'Bye'},{'role':'assistant','content':'Goodbye'}]}
  context,target=training_context(row);self.assertEqual(len(context),3);self.assertEqual(target,'Goodbye')
if __name__=='__main__':unittest.main()
