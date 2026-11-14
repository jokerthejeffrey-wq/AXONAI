"""Exercise durable upload retries without network or torch."""
import hashlib,tempfile,unittest
from pathlib import Path
from trainer.outbox import save_checkpoint,flush_pending

JOB={'id':'a'*32,'upload_token':'b'*64}
SERVER='https://axon.example'
DATA=b'SAFE'+b'0'*2600000

class Reply:
    def __init__(self,status,body=None):
        self.status_code=status
        self.body=body or {}
    def json(self):
        return self.body

class FakeSession:
    def __init__(self,failures=None,final_lost=False):
        self.failures=list(failures or [])
        self.calls=[]
        self.final_lost=final_lost
    def request(self,method,url,**kw):
        self.calls.append(url)
        if url.endswith('/api/job-state'):
            return Reply(200,{'status':'pending','sha256':hashlib.sha256(DATA).hexdigest(),'size':len(DATA)})
        if '/api/update-part' in url:
            if self.failures:
                return Reply(self.failures.pop(0),{'error':'storage temporarily unavailable'})
            return Reply(200)
        if url.endswith('/api/update'):
            return Reply(409 if self.final_lost else 200)
        raise AssertionError('Unexpected URL: '+url)

class OutboxTests(unittest.TestCase):
    def test_checkpoint_saved_before_network_and_uploaded_after_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            p=save_checkpoint(directory,JOB,SERVER,DATA)
            self.assertTrue(p.is_file())
            self.assertTrue(Path(str(p).replace('.json','.safetensors')).is_file())
            session=FakeSession([507])
            waits=[]
            sent=flush_pending(directory,session,SERVER,sleep=waits.append,log=lambda *a,**kw:None)
            self.assertEqual(sent,1)
            self.assertEqual(waits,[150])
            self.assertFalse(p.exists())
            self.assertTrue(len(session.calls)>=4)
    def test_final_response_lost_reconciles_via_authorized_state(self):
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory,JOB,SERVER,DATA)
            session=FakeSession(final_lost=True)
            self.assertEqual(flush_pending(directory,session,SERVER,log=lambda *a,**kw:None),1)
            self.assertIn(SERVER+'/api/job-state',session.calls)
    def test_wrong_server_never_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory,JOB,SERVER,DATA)
            session=FakeSession()
            with self.assertRaisesRegex(RuntimeError,'different coordinator'):
                flush_pending(directory,session,'https://other.example',log=lambda *a,**kw:None)
            self.assertFalse(session.calls)
    def test_corrupt_saved_checkpoint_never_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            p=save_checkpoint(directory,JOB,SERVER,DATA)
            Path(str(p).replace('.json','.safetensors')).write_bytes(b'corrupt')
            session=FakeSession()
            with self.assertRaisesRegex(RuntimeError,'SHA-256'):
                flush_pending(directory,session,SERVER,log=lambda *a,**kw:None)
            self.assertFalse(session.calls)

if __name__=='__main__':unittest.main()
