"""Product attribution is explicit metadata, not a model's guessed biography."""
import re
COMMUNITY='AXONAI was developed with the help of many people in its community, including developers, testers, and people contributing their computing power.'
def identity_answer(text,mode):
    text=' '.join(re.sub(r'[^\w\s]',' ',text.casefold()).split())
    patterns=[r'who.*\b(created|create|made|make|built|build|developed|develop|trained|train)\b.*\b(you|axonai)\b',r'\b(your|axonai s)\s+(creator|creators|developer|developers)\b',r'\bwho (are you|is behind axonai)\b',r'\b(seni kim|kim geliştirdi|kim yarattı)\b']
    if not any(re.search(p,text) for p in patterns):return None
    suffix=(' This chat currently uses a pretrained Qwen model while our independent student model is being trained.' if mode=='assisted' else ' This chat is running our independently trained student model.')
    return COMMUNITY+suffix
