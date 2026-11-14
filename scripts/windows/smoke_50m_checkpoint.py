"""Check full 50M safetensors export against the real AXON v0 checkpoint."""
import hashlib
import tempfile
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from safetensors import safe_open
from trainer.expanded_model import create_checkpoint
from trainer.eval_50m import validate_checkpoint
from settings import LARGE_CONFIG, parameter_count
paths=sorted((root/'public/model').glob('v0-*.bin'))
if len(paths)!=6:raise SystemExit('Expected six signed legacy model parts')
base=b''.join(p.read_bytes() for p in paths)
if hashlib.sha256(base).hexdigest()!='b7951bc8bc3c8187ae54b79653573e022a4cbac35454de6f8bbee91365b388f2':
    raise SystemExit('Initial checkpoint SHA256 mismatch')
with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp)
    legacy=temp/'small.safetensors'
    legacy.write_bytes(base)
    out=temp/'50m.safetensors'
    info=create_checkpoint(legacy,out)
    if info['parameter_count']!=50_224_640 or not 190_000_000<info['bytes']<210_000_000:
        raise SystemExit('Invalid 50M checkpoint size')
    with safe_open(str(out),framework='pt',device='cpu') as f:
        if list(f.get_tensor('token.weight').shape)!=[260,640]:
            raise SystemExit('Incorrect token embedding shape')
        if list(f.get_tensor('blocks.9.qkv.weight').shape)!=[1920,640]:
            raise SystemExit('Incorrect final attention shape')
        if list(f.get_tensor('position.weight').shape)!=[1024,640]:
            raise SystemExit('Incorrect position embedding')
    metrics=validate_checkpoint(out)
    if metrics['parameters']!=info['parameter_count'] or not 0 < metrics['held_out_loss'] < 200:
        raise SystemExit('Invalid independent validation loss')
    if metrics['conversational_readiness']!='not assessed':
        raise SystemExit('Unproven conversational capability must not be claimed')
    print('Verified expanded checkpoint:',
          f"{info['parameter_count']:,} parameters",
          f"{info['bytes']/1e6:.1f} MB",
          'held-out loss', f"{metrics['held_out_loss']:.3f}")
