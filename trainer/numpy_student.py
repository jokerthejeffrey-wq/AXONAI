"""Checkpoint-compatible transformer inference and output-layer training."""
import numpy as np
from safetensors.numpy import load_file, save_file
from settings import CONFIG

PAD, USER, ASSISTANT, END = 256, 257, 258, 259

class Student:
    def __init__(self, device='cpu'):
        self.config = CONFIG
        self.device = device
        if device == 'cuda':
            try:
                import cupy as cp
                cp.cuda.runtime.getDeviceCount()
                self.xp = cp
            except Exception as error:
                raise RuntimeError('GPU training needs an NVIDIA CUDA 12 driver. Use CPU if unavailable.') from error
        else:
            self.xp = np
        self.weights = None

    def load(self, path):
        self.weights = {k:self.xp.asarray(v) for k,v in load_file(str(path)).items()}
        if any(not bool(self.xp.isfinite(v).all()) for v in self.weights.values()):
            raise ValueError('Checkpoint contains invalid numbers')

    def save(self, path):
        if self.device == 'cuda':
            weights = {k:self.xp.asnumpy(v) for k,v in self.weights.items()}
        else:
            weights = self.weights
        save_file({k:np.ascontiguousarray(v) for k,v in weights.items()}, str(path))

    def features(self, ids):
        xp, w = self.xp, self.weights
        ids = xp.asarray(ids, dtype=xp.int32)
        length = len(ids)
        x = w['token.weight'][ids] + w['position.weight'][:length]
        def norm(x, name):
            mean=x.mean(axis=-1, keepdims=True)
            variance=((x-mean)**2).mean(axis=-1, keepdims=True)
            return (x-mean)/xp.sqrt(variance+1e-5)*w[name+'.weight']+w[name+'.bias']
        def linear(x, name):
            return x @ w[name+'.weight'].T + w[name+'.bias']
        for layer in range(self.config['layers']):
            p=f'blocks.{layer}.'
            qkv=linear(norm(x,p+'n1'),p+'qkv').reshape(length,3,8,32)
            q,k,v=(qkv[:,i].transpose(1,0,2) for i in range(3))
            scores=(q @ k.transpose(0,2,1)) / xp.sqrt(xp.float32(32))
            scores=xp.where(xp.triu(xp.ones((length,length),dtype=bool),1),-xp.inf,scores)
            scores=xp.exp(scores-scores.max(axis=-1,keepdims=True))
            attention=(scores/scores.sum(axis=-1,keepdims=True)) @ v
            x=x+linear(attention.transpose(1,0,2).reshape(length,256),p+'proj')
            hidden=linear(norm(x,p+'n2'),p+'ff.0')
            hidden=.5*hidden*(1+xp.tanh(xp.sqrt(xp.float32(2/np.pi))*(hidden+.044715*hidden**3)))
            x=x+linear(hidden,p+'ff.2')
        return norm(x,'norm')

    def train(self, rows, before_step=None):
        xp=self.xp
        for index,row in enumerate(rows):
            if before_step: before_step()
            prefix=[]
            for message in row['messages']:
                prefix += [USER if message['role']=='user' else ASSISTANT,*message['content'].encode(),END]
            prefix.append(ASSISTANT)
            target=[*row['response'].encode(),END]
            sequence=prefix+target
            if len(sequence)>self.config['context']+1: raise ValueError('Lesson exceeds context')
            features=self.features(sequence[:-1])[len(prefix)-1:]
            expected=xp.asarray(target,dtype=xp.int32)
            logits=features @ self.weights['head.weight'].T
            logits-=logits.max(axis=1,keepdims=True)
            probs=xp.exp(logits);probs/=probs.sum(axis=1,keepdims=True)
            loss=-xp.log(xp.maximum(probs[xp.arange(len(expected)),expected],1e-12)).mean()
            probs[xp.arange(len(expected)),expected]-=1
            grad=probs.T @ features / len(expected)
            norm=xp.linalg.norm(grad)
            grad*=xp.minimum(1,1/(norm+1e-6))
            self.weights['head.weight']-=xp.float32(3e-4)*grad
            print(f'Lesson {index+1}/{len(rows)} | loss {float(loss):.4f}',flush=True)
