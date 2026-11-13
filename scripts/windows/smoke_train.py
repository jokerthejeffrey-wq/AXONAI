"""Smoke test that bundled student.pyc performs full parameter backpropagation."""
import sys
from pathlib import Path

if len(sys.argv)!=2:
    raise SystemExit('Usage: smoke_train.py EMBEDDED_PAYLOAD_DIR')
payload=Path(sys.argv[1]).resolve()
if not (payload/'student.pyc').is_file():
    raise SystemExit('Compiled full-gradient model missing from EXE payload')
sys.path.insert(0,str(payload))
import torch
import student
from student import Student, train

if not str(Path(student.__file__).resolve()).startswith(str(payload)):
    raise SystemExit('Did not import packed student model')
torch.set_num_threads(2)
torch.manual_seed(42)
model=Student().to('cpu')
initial=model.blocks[0].qkv.weight.detach().clone()
train(model,[{
 'concept':'smoke',
 'messages':[{'role':'user','content':'hello'}],
 'response':'Hello.'
}], 'cpu')
after=model.blocks[0].qkv.weight.detach()
if torch.allclose(initial,after):
    raise SystemExit('Model output layer changed, but transformer weights did not')
if not torch.isfinite(after).all():
    raise SystemExit('Nonfinite model weights')
print('Full-gradient checkpoint training verified on CPU')
