"""Bounded, source-grounded lesson extraction for AXON's 512-token student.

Do not alter corpus.training_context: it defines the already-published corpus
record ordering and fingerprints. This module handles longer source dialogues
*after* the coordinator assigns a unique record.
"""
import re

_BAD=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
MAX_PROMPT_BYTES=380
MAX_CONTEXT_BYTES=509
MAX_SOURCE_BYTES=32768

def _valid(value):
    return (isinstance(value,str) and value.strip() and not _BAD.search(value)
            and len(value.encode('utf-8'))<=MAX_SOURCE_BYTES)

def _clip(text,limit):
    if len(text.encode('utf-8'))<=limit:
        return text.strip()
    excerpt=text.encode('utf-8')[:limit].decode('utf-8',errors='ignore')
    # Do not finish a truncated answer halfway through a word if avoidable.
    cut=excerpt.rfind(' ')
    if cut>limit//2:
        excerpt=excerpt[:cut]
    return excerpt.strip()

def source_pairs(raw):
    """Yield source-exact (question, answer) pairs; no invented completions."""
    messages=raw.get('messages') if isinstance(raw,dict) else None
    if not isinstance(messages,list):
        return []
    pairs=[]
    user=None
    for message in messages:
        if not isinstance(message,dict):
            user=None
            continue
        role=message.get('role')
        content=message.get('content')
        if role=='user':
            user=content if _valid(content) else None
        elif role=='assistant' and user is not None:
            if (_valid(content) and user.strip().casefold()!=content.strip().casefold()):
                pairs.append((user.strip(),content.strip()))
            user=None
    return pairs

def source_lessons(raw,max_lessons=3):
    """Up to three useful lessons per unique record; no repeated global data.

    A short exact Q/A is preferred. When its question is too long, use
    contiguous text *from its own assistant answer* as a language-continuation
    lesson, avoiding a mismatched cropped question and fabricated answer.
    """
    if not isinstance(max_lessons,int) or not 1<=max_lessons<=8:
        raise ValueError('Invalid lesson limit')
    prepared=[]
    pairs=source_pairs(raw)
    for question,answer in reversed(pairs[-3:]):
        if len(question.encode('utf-8'))<=MAX_PROMPT_BYTES:
            budget=MAX_CONTEXT_BYTES-len(question.encode('utf-8'))-2
            reply=_clip(answer,budget)
            if len(reply)>=2:
                prepared.append({'concept':'conversation',
                                 'messages':[{'role':'user','content':question}],
                                 'response':reply})
        # Long real answers can also teach natural word sequences, even if
        # their associated question would not fit the model's context.
        if len(prepared)<max_lessons and len(answer.encode('utf-8'))>=180:
            lead=_clip(answer,100)
            remainder=answer[len(lead):].lstrip()
            reply=_clip(remainder,220)
            prompt='Continue this passage:\n'+lead
            if len(reply)>=48 and len(prompt.encode('utf-8'))+len(reply.encode('utf-8'))+4<=512:
                prepared.append({'concept':'source-text-continuation',
                                 'messages':[{'role':'user','content':prompt}],
                                 'response':reply})
        if len(prepared)>=max_lessons:
            break
    return prepared[:max_lessons]
