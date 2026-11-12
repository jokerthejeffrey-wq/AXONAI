"""Assisted chat by default; explicit student-only mode after owner evaluation."""
if __name__=='__main__':
    from bootstrap import ensure
    ensure('teacher','CHAT.py')
import threading,os
from ledger import version,checkpoint
from identity import identity_answer
def chat_mode():
    mode=os.getenv("AXON_CHAT_MODE","assisted").lower()
    if mode not in ("assisted","student"):raise RuntimeError("AXON_CHAT_MODE must be assisted or student")
    return mode
class ChatEngine:
    def __init__(self):self.model=None;self.assistant=None;self.loaded_version=None;self.mode=chat_mode();self.lock=threading.Lock()
    def chat(self,messages):
        import torch
        from student import Student,load_weights,generate
        with self.lock:
            answer=identity_answer(messages[-1]['content'],self.mode)
            if answer:return answer
            if self.mode=='assisted':
                if self.assistant is None:
                    from assisted import AssistedModel
                    self.assistant=AssistedModel(os.getenv('AXON_CHAT_DEVICE','cpu'))
                return self.assistant.reply(messages)
            v=version()
            if v==0:raise RuntimeError('Student is untrained. Train and publish a checkpoint first; no teacher fallback is used.')
            if not checkpoint(v).exists():raise RuntimeError('Student checkpoint missing. Restore a backup.')
            if self.model is None:
                torch.set_num_threads(int(os.getenv('AXON_THREADS','2')))
                self.device=os.getenv('AXON_CHAT_DEVICE','cpu');self.model=Student().to(self.device)
            if self.loaded_version!=v:load_weights(self.model,checkpoint(v));self.loaded_version=v
            return generate(self.model,messages,self.device)
if __name__=='__main__':
    engine=ChatEngine();history=[]
    print('AXONAI | mode:',engine.mode,'| /new | /quit')
    while True:
        try:
            text=input('\nYou: ').strip()
            if text=='/quit':break
            if text=='/new':history.clear();continue
            if not text:continue
            outgoing=history[-12:]+[dict(role='user',content=text)]
            answer=engine.chat(outgoing);history=outgoing+[dict(role='assistant',content=answer)]
            print('\nAXONAI:',answer)
        except (EOFError,KeyboardInterrupt):break
        except RuntimeError as e:print(e)
