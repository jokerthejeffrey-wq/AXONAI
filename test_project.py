"""Offline tests: real student gradients; no teacher or model downloads required."""
import os,tempfile,unittest,io,json,hashlib,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
TEMP=tempfile.TemporaryDirectory();os.environ['AXON_STATE']=TEMP.name
import torch,numpy as np
from safetensors.numpy import save,save_file
from safetensors.torch import save_file as save_torch
import HTML,ledger
from ledger import connect,checkpoint
from student import Student,snapshot,train,load_weights,generate,tensors

class ProjectTests(unittest.TestCase):
    def setUp(self):
        HTML.rates.clear();os.environ.pop('AXON_TRAIN_TOKEN',None);os.environ.pop('AXON_ALLOWED_ORIGINS',None)
        with connect() as c:
            for table in ['assigned','jobs','updates','lessons','concepts']:c.execute('DELETE FROM '+table)
            c.execute("UPDATE meta SET value='0' WHERE key='version'")
        checkpoint(0).parent.mkdir(exist_ok=True)
        save_file({'test.weight':np.zeros((2,2),np.float32)},str(checkpoint(0)))
        self.client=HTML.app.test_client()
    def seed(self,concept='greetings',count=8):
        with connect() as c:
            c.execute('INSERT INTO concepts(name) VALUES(?)',(concept,))
            for i in range(count):
                row=dict(concept=concept,messages=[dict(role='user',content='Hello '+str(i))],response='Hi there.')
                c.execute('INSERT INTO lessons(id,concept,payload) VALUES(?,?,?)',(str(i),concept,json.dumps(row)))
    def test_greetings_share_four_uses(self):
        self.seed();r=self.client.post('/api/job');self.assertEqual(r.status_code,200);self.assertEqual(len(r.json['rows']),4)
        self.assertEqual(self.client.post('/api/job').status_code,204)
        with connect() as c:self.assertEqual(c.execute('SELECT uses FROM concepts').fetchone()[0],4)
    def test_parallel_jobs_cannot_exceed_budget(self):
        self.seed()
        def take(_):
            with HTML.app.test_client() as client:
                r=client.post('/api/job');return len(r.json['rows']) if r.status_code==200 else 0
        with ThreadPoolExecutor(max_workers=4) as pool:self.assertEqual(sum(pool.map(take,range(4))),4)
    def test_reimport_and_answer_edit_do_not_reset_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir();file=root/'data/train.jsonl'
            row=dict(concept='hello',messages=[dict(role='user',content='Hello!')],response='Hi.')
            file.write_text(json.dumps(row)+'\n')
            with patch.object(ledger,'ROOT',root):
                ledger.import_lessons();self.client.post('/api/job')
                row['response']='A changed answer';file.write_text(json.dumps(row)+'\n');ledger.import_lessons()
            self.assertEqual(self.client.post('/api/job').status_code,204)
            with connect() as c:self.assertEqual(c.execute('SELECT count(*) FROM lessons').fetchone()[0],1)
    def test_upload_replay_and_stale(self):
        self.seed();job=self.client.post('/api/job').json
        def submit():return self.client.post('/api/update',data={'job':job['id'],'weights':(io.BytesIO(save({'test.weight':np.full((2,2),.01,np.float32)})),'student.safetensors')})
        self.assertEqual(submit().status_code,201);self.assertEqual(submit().status_code,409)
        with connect() as c:c.execute('UPDATE jobs SET used=0');c.execute("UPDATE meta SET value='1' WHERE key='version'")
        self.assertEqual(submit().status_code,409)
    def test_bad_weights(self):
        self.seed();job=self.client.post('/api/job').json
        for weights in [{'wrong':np.zeros(1,np.float32)},{'test.weight':np.full((2,2),np.nan,np.float32)},{'test.weight':np.ones((2,2),np.float32)}]:
            r=self.client.post('/api/update',data={'job':job['id'],'weights':(io.BytesIO(save(weights)),'x')});self.assertEqual(r.status_code,400)
    def test_cors_and_downloads(self):
        os.environ['AXON_ALLOWED_ORIGINS']='https://axon-demo.vercel.app'
        r=self.client.options('/api/chat',headers={'Origin':'https://axon-demo.vercel.app','Access-Control-Request-Method':'POST'})
        self.assertEqual(r.status_code,204);self.assertEqual(r.headers['Access-Control-Allow-Origin'],'https://axon-demo.vercel.app')
        self.assertEqual(self.client.get('/api/status',headers={'Origin':'https://bad.example'}).status_code,403)
        for path in ['/','/chat','/docs','/config.js','/download/client.zip','/download/source/student.py']:
            with self.client.get(path) as response:self.assertEqual(response.status_code,200,path)
        self.assertEqual(self.client.get('/download/source/TEACHER.py').status_code,404)
    def test_no_untrained_teacher_fallback(self):
        from CHAT import ChatEngine
        with patch.dict(os.environ,{'AXON_CHAT_MODE':'student'}):
            with self.assertRaisesRegex(RuntimeError,'untrained'):ChatEngine().chat([dict(role='user',content='Hello')])
        self.assertNotIn('transformers',sys.modules)
    def test_student_gradients_roundtrip_and_prompt_mask(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        config=dict(vocab_size=260,width=32,layers=2,heads=4,context=512)
        m=Student(config);before=snapshot(m)
        row=dict(concept='greetings',messages=[dict(role='user',content='Hi')],response='Hello!')
        x,y=tensors(row,'cpu');self.assertEqual(int((y!=-100).sum()),len('Hello!')+1)
        train(m,[row],'cpu');after=snapshot(m)
        self.assertTrue(any(not torch.equal(before[k],after[k]) for k in before))
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'x.safetensors';save_torch(after,str(file));other=Student(config);load_weights(other,file)
            self.assertTrue(all(torch.equal(after[k],snapshot(other)[k]) for k in after))
        self.assertIsInstance(generate(m,row['messages'],max_tokens=5),str)
        self.assertNotIn('transformers',sys.modules)
    def test_causal_attention(self):
        torch.manual_seed(2);m=Student(dict(vocab_size=260,width=32,layers=2,heads=4,context=16)).eval()
        with torch.no_grad():
            a=m(torch.tensor([[1,2,3,4]]));b=m(torch.tensor([[1,2,9,9]]))
        self.assertTrue(torch.allclose(a[:,:2],b[:,:2],atol=1e-6))
    def test_client_integrity_manifest(self):
        from settings import ROOT
        manifest=json.loads((ROOT/'client_manifest.json').read_text())
        for name,digest in manifest.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
    def test_source_repair_and_hash_rejection(self):
        import bootstrap
        from settings import ROOT
        manifest=json.loads((ROOT/'client_manifest.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in manifest:(root/name).write_bytes((ROOT/name).read_bytes())
            (root/'client_manifest.json').write_text(json.dumps(manifest))
            original=(root/'client.py').read_bytes();(root/'client.py').write_text('broken')
            with patch.object(bootstrap,'ROOT',root),patch('builtins.input',side_effect=['y','https://trusted.example']),patch.object(bootstrap,'urlopen',return_value=io.BytesIO(original)),patch.dict(os.environ,{'AXON_SERVER':''}),patch.object(sys,'argv',['CPU.py']):
                with self.assertRaises(SystemExit):bootstrap.repair_sources()
            self.assertEqual((root/'client.py').read_bytes(),original)
            (root/'client.py').write_text('broken')
            with patch.object(bootstrap,'ROOT',root),patch('builtins.input',side_effect=['y','https://trusted.example']),patch.object(bootstrap,'urlopen',return_value=io.BytesIO(b'wrong download')),patch.dict(os.environ,{'AXON_SERVER':''}),patch.object(sys,'argv',['CPU.py']):
                with self.assertRaisesRegex(SystemExit,'hash mismatch'):bootstrap.repair_sources()
            self.assertEqual((root/'client.py').read_text(),'broken')

    def test_community_identity_and_assisted_routing(self):
        from CHAT import ChatEngine
        from identity import identity_answer
        for question in ['who did create you','Who created you?','Who made you?','Who are your creators?']:
            answer=identity_answer(question,'assisted')
            self.assertIn('community',answer);self.assertIn('Qwen',answer);self.assertNotIn('Anthropic',answer)
        with patch.dict(os.environ,{'AXON_CHAT_MODE':'assisted'}):
            engine=ChatEngine()
            class Stub:
                def reply(self,messages):return 'Assisted test reply'
            engine.assistant=Stub()
            self.assertEqual(engine.chat([dict(role='user',content='Explain gravity')]),'Assisted test reply')
            self.assertIsNone(engine.model)
            self.assertIsNone(engine.loaded_version)

    def test_auth(self):
        os.environ['AXON_TRAIN_TOKEN']='secret'
        self.assertEqual(self.client.post('/api/job').status_code,401)
        self.seed();self.assertEqual(self.client.post('/api/job',headers={'Authorization':'Bearer secret'}).status_code,200)

if __name__=='__main__':unittest.main()
