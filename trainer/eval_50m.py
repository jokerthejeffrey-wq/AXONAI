"""Held-out language loss for 50M and legacy AXON checkpoints.

This measures prediction of reference answers on a tiny, fixed validation
sample. It does NOT certify conversational understanding.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from settings import CONFIG, LARGE_CONFIG
from student import Student, evaluate, load_weights


def checkpoint_profile(path):
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        w=list(reader.get_tensor('token.weight').shape)
        positions=list(reader.get_tensor('position.weight').shape)
        if w==[CONFIG['vocab_size'],CONFIG['width']] and positions==[CONFIG['context'],CONFIG['width']]:
            return dict(CONFIG)
        if w==[LARGE_CONFIG['vocab_size'],LARGE_CONFIG['width']] and positions==[LARGE_CONFIG['context'],LARGE_CONFIG['width']]:
            return dict(LARGE_CONFIG)
    raise ValueError('Unsupported AXON checkpoint; no generation attempted')


def validation_rows(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                value=json.loads(line)
                if (not isinstance(value.get('messages'),list) or
                    not isinstance(value.get('response'),str) or
                    not value['response'].strip()):
                    raise ValueError('Invalid held-out example')
                rows.append(value)
    if not rows:
        raise ValueError('No held-out validation data')
    return rows


def validate_checkpoint(path, *, device='cpu', validation=None):
    config=checkpoint_profile(path)
    if validation is None:
        validation=Path(__file__).resolve().parents[1]/'data/validation.jsonl'
    rows=validation_rows(validation)
    model=Student(config).to(device)
    load_weights(model,path)
    loss=evaluate(model,rows,device)
    if not math.isfinite(loss):
        raise ValueError('Invalid model validation result')
    return {'parameters':sum(p.numel() for p in model.parameters()),
            'validation_examples':len(rows),
            'held_out_loss':loss,
            'perplexity':math.exp(min(50,loss)),
            'conversational_readiness':'not assessed'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    scores=validate_checkpoint(args.checkpoint,device=args.device)
    print(json.dumps(scores,indent=2))
    print('A lower held-out loss is preferable, but does not prove that a model '
          'can hold a normal conversation.')


if __name__=='__main__':
    main()
