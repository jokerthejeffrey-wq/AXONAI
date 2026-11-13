import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from trainer.memory import MB, budgets, wait, MemoryAdmissionError

class Clock:
    def __init__(self):
        self.now=0
    def time(self):
        return self.now
    def sleep(self,s):
        self.now+=s

class MemoryTests(unittest.TestCase):
    def mem(self, total_mb,used_mb):
        return SimpleNamespace(total=total_mb*MB, available=(total_mb-used_mb)*MB)

    def test_allows_allocation_with_90_percent_headroom(self):
        # 8 GB with 5 GB in use has about 2.2 GB available before 90%.
        with patch('trainer.memory.psutil.virtual_memory',return_value=self.mem(8192,5120)):
            state=budgets(1100)
            self.assertTrue(state[0]['allowed'])
            wait(1100,max_wait_seconds=5,output=io.StringIO())

    def test_ram_budget_is_not_disabled(self):
        with patch('trainer.memory.psutil.virtual_memory',return_value=self.mem(8192,7600)):
            self.assertFalse(budgets(1100)[0]['allowed'])

    def test_timeout_does_not_spam_or_wait_forever(self):
        c=Clock()
        output=io.StringIO()
        with patch('trainer.memory.psutil.virtual_memory',return_value=self.mem(8192,7600)):
            with self.assertRaisesRegex(MemoryAdmissionError,'Training stopped'):
                wait(1100,max_wait_seconds=10,interval=5,clock=c.time,sleep=c.sleep,output=output)
        self.assertEqual(c.now,10)
        self.assertLessEqual(output.getvalue().count('\\n'),1)
        self.assertIn('safe free',output.getvalue())
        self.assertIn('1100 MB',output.getvalue())

    def test_training_resumes_if_ram_becomes_available(self):
        c=Clock()
        output=io.StringIO()
        values=[self.mem(8192,7600),self.mem(8192,7600),self.mem(8192,5200)]
        def vm():
            return values.pop(0)
        with patch('trainer.memory.psutil.virtual_memory',side_effect=vm):
            wait(1100,max_wait_seconds=20,interval=5,clock=c.time,sleep=c.sleep,output=output)
        self.assertEqual(c.now,10)
        self.assertIn('RAM available; continuing.',output.getvalue())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            budgets(-100)
        with self.assertRaises(ValueError):
            wait(200,max_wait_seconds=-1)
if __name__=='__main__':
    unittest.main()
