from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
STATE=Path(os.getenv('AXON_STATE',str(ROOT/'student_state'))).resolve()
CONFIG=dict(vocab_size=260,width=256,layers=4,heads=8,context=512)
# Research track: 50,224,640 trainable parameters. This is not the
# existing production checkpoint layout; do not point old /api/job at it.
LARGE_CONFIG=dict(vocab_size=260,width=640,layers=10,heads=8,context=1024)
TRAINING_PROFILES={'legacy':CONFIG,'axon50m':LARGE_CONFIG}

def parameter_count(config):
    d,v,c,l=(config['width'],config['vocab_size'],config['context'],config['layers'])
    return (l*(12*d*d+13*d) + 2*v*d + c*d + 2*d)

TEACHER='Qwen/Qwen2.5-0.5B-Instruct'
TEACHER_REVISION='7ae5576'
MAX_USES=4
BATCH_LESSONS=8

def manifest(): return dict(protocol=2,architecture=CONFIG,tokenizer='utf8-bytes-v1',max_uses=MAX_USES)
