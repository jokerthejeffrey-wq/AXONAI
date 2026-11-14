"""Randomly initialized causal transformer. No pretrained model/tokenizer imports."""
import torch
from torch import nn
import torch.nn.functional as F
from settings import CONFIG
PAD,USER,ASSISTANT,END=256,257,258,259

def encode(text): return list(text.encode('utf-8'))
def conversation(messages):
    result=[]
    for m in messages:
        result += [USER if m['role']=='user' else ASSISTANT]+encode(m['content'])+[END]
    return result

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['width'];self.heads=c['heads']
        self.n1=nn.LayerNorm(d);self.qkv=nn.Linear(d,d*3);self.proj=nn.Linear(d,d)
        self.n2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(a.transpose(1,2).reshape(b,t,d));return x+self.ff(self.n2(x))

    def incremental(self,x,previous=None):
        """Inference-only attention with a per-conversation key/value cache."""
        b,t,d=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        if previous is not None:
            if t != 1: raise ValueError('Cached decoding accepts one token at a time')
            k=torch.cat((previous[0],k),dim=2)
            v=torch.cat((previous[1],v),dim=2)
        # The sole new query can attend to all preceding keys and itself.
        a=F.scaled_dot_product_attention(q,k,v,is_causal=previous is None)
        x=x+self.proj(a.transpose(1,2).reshape(b,t,d))
        return x+self.ff(self.n2(x)), (k,v)

class Student(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or CONFIG;c=self.config
        self.token=nn.Embedding(c['vocab_size'],c['width']);self.position=nn.Embedding(c['context'],c['width'])
        self.blocks=nn.Sequential(*[Block(c) for _ in range(c['layers'])]);self.norm=nn.LayerNorm(c['width'])
        self.head=nn.Linear(c['width'],c['vocab_size'],bias=False)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,ids):
        if ids.shape[1]>self.config['context']: raise ValueError('Context too long')
        x=self.token(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        return self.head(self.norm(self.blocks(x)))

    @torch.inference_mode()
    def incremental(self,ids,cache=None):
        position=cache[0][0].shape[2] if cache else 0
        if position+ids.shape[1]>self.config['context']: raise ValueError('Context too long')
        x=self.token(ids)+self.position(torch.arange(position,position+ids.shape[1],device=ids.device))
        updated=[]
        for i,block in enumerate(self.blocks):
            x,kv=block.incremental(x,cache[i] if cache else None)
            updated.append(kv)
        return self.head(self.norm(x)),updated

def snapshot(model): return {k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}

def load_weights(model,path):
    from safetensors.torch import load_file
    incoming=load_file(str(path)); expected=model.state_dict()
    if incoming.keys()!=expected.keys(): raise ValueError('Wrong checkpoint keys')
    for k,t in incoming.items():
        if t.shape!=expected[k].shape or t.dtype!=torch.float32 or not torch.isfinite(t).all(): raise ValueError('Invalid tensor')
    model.load_state_dict(incoming,strict=True)

def tensors(row,device,context=None):
    context=context or CONFIG['context']
    prefix=conversation(row['messages'])+[ASSISTANT]
    target=encode(row['response'])+[END]
    if len(prefix)+len(target)>context+1: raise ValueError('Lesson too long: shorten context or response')
    seq=prefix+target
    x=torch.tensor([seq[:-1]],device=device)
    y=torch.tensor([seq[1:]],device=device);y[:,:len(prefix)-1]=-100
    return x,y

def train(model,rows,device,before_step=None):
    """Exactly ONE optimizer step per assigned lesson; no epochs/replacement sampling."""
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    for i,row in enumerate(rows):
        if before_step:before_step()
        x,y=tensors(row,device,model.config['context']);opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x).reshape(-1,260),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss; update not uploaded')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        print(f"Lesson {i+1}/{len(rows)} | {row['concept']} | loss {loss.item():.4f}",flush=True)
    model.eval()

@torch.inference_mode()
def evaluate(model,rows,device='cpu'):
    if not rows:raise ValueError('Separate validation lessons required')
    model.eval();losses=[]
    for row in rows:
        x,y=tensors(row,device,model.config['context'])
        losses.append(F.cross_entropy(model(x).reshape(-1,260),y.reshape(-1),ignore_index=-100).item())
    return sum(losses)/len(losses)

def generate_stream(model,messages,device='cpu',max_tokens=240,temperature=.8,top_p=.9):
    """Yield UTF-8 text incrementally from student probabilities, using KV caching.

    Tokens are bytes, not whole words. Incomplete Unicode sequences are buffered.
    This improves inference efficiency; it does not itself teach reasoning.
    """
    import codecs
    import math
    if not math.isfinite(temperature) or temperature<=0 or not 0<top_p<=1:
        raise ValueError('Invalid sampling settings')
    if not isinstance(max_tokens,int) or max_tokens<1: raise ValueError('Invalid token budget')
    model.eval()
    context=model.config['context']
    # Reserve output space while keeping the end of the latest conversation.
    max_tokens=min(max_tokens,context-1)
    ids=(conversation(messages)+[ASSISTANT])[-(context-max_tokens):]
    out=[];cache=None
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    for _ in range(max_tokens):
        with torch.inference_mode():
            logits,cache=model.incremental(torch.tensor([ids],device=device),cache)
            logits=logits[0,-1].float().clone()
            logits[PAD]=logits[USER]=logits[ASSISTANT]=-float('inf')
            for token in set(out[-64:]):
                logits[token]=logits[token]/1.08 if logits[token]>0 else logits[token]*1.08
            values,indices=torch.topk(logits/temperature,min(40,len(logits)))
            probs=values.softmax(-1)
            remove=probs.cumsum(-1)-probs>=top_p
            probs[remove]=0
            token=indices[torch.multinomial(probs,1)].item()
        if token==END: break
        out.append(token);ids=[token]
        text=decoder.decode(bytes([token]),final=False)
        if text: yield text
    tail=decoder.decode(b'',final=True)
    if tail: yield tail


def generate(model,messages,device='cpu',max_tokens=240,temperature=.8):
    return ''.join(generate_stream(model,messages,device,max_tokens,temperature)).strip()
