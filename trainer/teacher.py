"""Tiny local lesson helper. NO AI model, downloaded weights, Transformers, or network.
This deterministic teacher performs source-grounded quality screening and
conservative instruction variation. It cannot fact-check unknown claims.
"""
import re
from trainer.lessons import source_pairs, source_lessons

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
        # Quality and bounded-context selection use the same source rules as
        # the trainer. Reject corrupted text, not merely long dialogues.
        return bool(source_lessons(raw))

    def generate_checked(self, raw, assignment_seed):
        # Preserve the ORIGINAL factual answer. Do not invent facts, numbers,
        # or paraphrases that could silently alter its meaning.
        if assignment_seed % 4 != 0 or not self.review_source(raw):
            return None
        pairs = source_pairs(raw)
        if not pairs:
            return None
        question, answer = pairs[-1]
        # Conservative question rephrasing uses an original question/answer.
        # It never fabricates an answer or substitutes model-generated facts.
        if not 8 <= len(question) <= 140:
            return None
        if any(question.casefold().startswith(p.casefold()) for p in _PREFIXES):
            return None
        changed=[{'role':'user',
                  'content':_PREFIXES[(assignment_seed//4)%len(_PREFIXES)]+question}]
        # The live checkpoint remains limited to 512 UTF-8 byte tokens.
        length=sum(len(m['content'].encode('utf-8'))+2 for m in changed)+len(answer.encode('utf-8'))+2
        if length>510:
            return None
        return {'concept':'local-source-grounded-variation','messages':changed,'response':answer}

    def close(self):
        pass
