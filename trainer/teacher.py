"""Tiny local lesson helper. NO AI model, downloaded weights, Transformers, or network.
This deterministic teacher performs source-grounded quality screening and
conservative instruction variation. It cannot fact-check unknown claims.
"""
import re
from trainer.corpus import training_context

_BAD = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_PREFIXES = (
    'Please answer this question: ',
    'Can you explain the following? ',
    'Give a clear response to this: ',
    'Please help with this question: ',
)

class Teacher:
    def __init__(self, device='cpu'):
        print('Local lesson helper ready (standard library; zero pretrained downloads).', flush=True)

    def review_source(self, raw):
        context, answer = training_context(raw)
        if not context or not answer or not isinstance(answer,str):
            return False
        if context[-1]['role'] != 'user':
            return False
        # Gate 1: ensure UTF-8 lengths and structure are bounded.
        # The trainer truncates long answers to its 512-token context window.
        # Do not discard otherwise valid examples solely for being >700 bytes.
        if len(answer.encode('utf-8')) > 32768 or len(answer.strip()) < 2:
            return False
        # Gate 2: reject control characters and blank/duplicate content.
        texts=[m.get('content','') for m in context]
        return (all(isinstance(t,str) and t.strip() and not _BAD.search(t) for t in texts)
                and not _BAD.search(answer) and answer.strip().casefold()!=texts[-1].strip().casefold())

    def generate_checked(self, raw, assignment_seed):
        # Preserve the ORIGINAL factual answer. Do not invent facts, numbers,
        # or paraphrases that could silently alter its meaning.
        if assignment_seed % 4 != 0 or not self.review_source(raw):
            return None
        context, answer = training_context(raw)
        if len(context)>6:
            return None
        question=context[-1]['content'].strip()
        if not 8 <= len(question) <= 140:
            return None
        if any(question.casefold().startswith(p.casefold()) for p in _PREFIXES):
            return None
        changed=[dict(m) for m in context]
        changed[-1]['content']=_PREFIXES[(assignment_seed//4)%len(_PREFIXES)]+question
        # The live checkpoint remains limited to 512 UTF-8 byte tokens.
        length=sum(len(m['content'].encode('utf-8'))+2 for m in changed)+len(answer.encode('utf-8'))+2
        if length>510:
            return None
        return {'concept':'local-source-grounded-variation','messages':changed,'response':answer}

    def close(self):
        pass
