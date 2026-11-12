from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
STATE=Path(os.getenv('AXON_STATE',str(ROOT/'student_state'))).resolve()
CONFIG=dict(vocab_size=260,width=256,layers=4,heads=8,context=512)
TEACHER='Qwen/Qwen2.5-0.5B-Instruct'
TEACHER_REVISION='7ae5576'
MAX_USES=4
BATCH_LESSONS=8

def manifest(): return dict(protocol=2,architecture=CONFIG,tokenizer='utf8-bytes-v1',max_uses=MAX_USES)
