import unittest
from types import SimpleNamespace
from trainer.assignments import rounds, claim_until_ready

class ContinuousTrainingTests(unittest.TestCase):
    def response(self,status,reason=None,retry=None):
        headers={}
        if reason:headers['X-AXON-Queue']=reason
        if retry:headers['Retry-After']=retry
        return SimpleNamespace(status_code=status,headers=headers)

    def test_unlimited_rounds_by_default(self):
        stream=rounds()
        self.assertEqual([next(stream) for _ in range(3000)],list(range(3000)))
        self.assertEqual(list(rounds(3)),[0,1,2])
        with self.assertRaises(ValueError):rounds(-1)

    def test_queue_and_legacy_daily_state_do_not_stop(self):
        items=iter([self.response(204,'review-pending','45'),
                    self.response(204,'worker-daily-limit','60'),
                    self.response(204,'daily-capacity-reached','60'),
                    self.response(200)])
        waits=[]
        reply=claim_until_ready(lambda:next(items),sleep=waits.append,printer=lambda *a,**kw:None)
        self.assertEqual(reply.status_code,200)
        self.assertEqual(waits,[45,60,60])

    def test_corpus_exhaustion_stops_without_replaying(self):
        waits=[]
        result=claim_until_ready(lambda:self.response(204,'corpus-complete'),sleep=waits.append,printer=lambda *a,**kw:None)
        self.assertIsNone(result)
        self.assertEqual(waits,[])

    def test_wait_is_bounded(self):
        items=iter([self.response(204,'review-pending','9999'),self.response(204,'review-pending','0'),self.response(200)])
        waits=[]
        claim_until_ready(lambda:next(items),sleep=waits.append,printer=lambda *a,**kw:None)
        self.assertEqual(waits,[300,15])

if __name__=='__main__': unittest.main()
