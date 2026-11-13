// AXON's from-scratch byte-level transformer inference; no pretrained model API.
import {TextDecoder} from 'node:util';
const W=256,H=8,D=32,V=260,C=512,L=4,END=259,USER=257,ASSISTANT=258,PAD=256;
const decoder=new TextDecoder('utf-8',{fatal:true});
export function parseCheckpoint(buffer){
  if(!Buffer.isBuffer(buffer)||buffer.length<10000||buffer.length>32*1024*1024)throw Error('Invalid model length');
  const n=Number(buffer.readBigUInt64LE(0));
  if(!Number.isSafeInteger(n)||n<=0||n>100000||n+8>buffer.length)throw Error('Invalid checkpoint header');
  const meta=JSON.parse(buffer.subarray(8,8+n).toString('utf8')),dataOffset=8+n,entries=new Map();
  for(const [key,v] of Object.entries(meta)){
    if(key==='__metadata__')continue;
    if(!/^(?:token|position|head|norm|blocks\.[0-3]\.(?:n1|n2|qkv|proj|ff\.[02]))\.(?:weight|bias)$/.test(key))throw Error('Unexpected weight');
    if(v.dtype!=='F32'||!Array.isArray(v.data_offsets)||!Array.isArray(v.shape))throw Error('Invalid weight');
    const [a,b]=v.data_offsets,size=v.shape.reduce((p,q)=>p*q,1);
    if(!Number.isSafeInteger(size)||size<=0||b-a!==size*4||a<0||b+dataOffset>buffer.length)throw Error('Invalid shape');
    const copy=Buffer.from(buffer.subarray(dataOffset+a,dataOffset+b));
    entries.set(key,{values:new Float32Array(copy.buffer,copy.byteOffset,size),shape:v.shape});
  }
  const tensor=(k,shape)=>{
    const v=entries.get(k);
    if(!v||v.shape.join(',')!==shape.join(','))throw Error('Model shape mismatch');
    return v.values;
  };
  const layers=[];
  for(let i=0;i<L;i++){
    const p='blocks.'+i+'.';
    layers.push({
      n1w:tensor(p+'n1.weight',[W]),n1b:tensor(p+'n1.bias',[W]),
      qkvW:tensor(p+'qkv.weight',[W*3,W]),qkvB:tensor(p+'qkv.bias',[W*3]),
      projW:tensor(p+'proj.weight',[W,W]),projB:tensor(p+'proj.bias',[W]),
      n2w:tensor(p+'n2.weight',[W]),n2b:tensor(p+'n2.bias',[W]),
      ff1W:tensor(p+'ff.0.weight',[4*W,W]),ff1B:tensor(p+'ff.0.bias',[4*W]),
      ff2W:tensor(p+'ff.2.weight',[W,4*W]),ff2B:tensor(p+'ff.2.bias',[W])
    });
  }
  if(entries.size!==L*12+5)throw Error('Incorrect checkpoint tensor count');
  return {token:tensor('token.weight',[V,W]),position:tensor('position.weight',[C,W]),
    normW:tensor('norm.weight',[W]),normB:tensor('norm.bias',[W]),
    head:tensor('head.weight',[V,W]),layers};
}
function norm(v,g,b){
  let mean=0,variance=0;
  for(let i=0;i<v.length;i++)mean+=v[i];
  mean/=v.length;
  for(let i=0;i<v.length;i++)variance+=(v[i]-mean)**2;
  const scale=1/Math.sqrt(variance/v.length+1e-5),out=new Float32Array(v.length);
  for(let i=0;i<v.length;i++)out[i]=(v[i]-mean)*scale*g[i]+b[i];
  return out;
}
function linear(v,weights,bias,rows){
  const out=new Float32Array(rows),cols=v.length;
  for(let r=0;r<rows;r++){
    let total=bias?bias[r]:0,offset=r*cols;
    for(let c=0;c<cols;c++)total+=weights[offset+c]*v[c];
    out[r]=total;
  }
  return out;
}
function erf(x){
  const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+.3275911*a);
  return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-a*a));
}
const gelu=x=>x*.5*(1+erf(x/Math.SQRT2));
function add(x,y){const o=new Float32Array(x.length);for(let i=0;i<o.length;i++)o[i]=x[i]+y[i];return o;}
function logitsForToken(m,id,cache){
  const pos=cache[0].k.length;
  if(pos>=C||!Number.isInteger(id)||id<0||id>=V)throw Error('Model context limit');
  let x=new Float32Array(W);
  for(let i=0;i<W;i++)x[i]=m.token[id*W+i]+m.position[pos*W+i];
  for(let layer=0;layer<L;layer++){
    const p=m.layers[layer],mem=cache[layer],qkv=linear(norm(x,p.n1w,p.n1b),p.qkvW,p.qkvB,W*3);
    const q=qkv.subarray(0,W);
    mem.k.push(Float32Array.from(qkv.subarray(W,2*W)));
    mem.v.push(Float32Array.from(qkv.subarray(2*W,3*W)));
    const attention=new Float32Array(W),n=mem.k.length;
    for(let h=0;h<H;h++){
      const offset=h*D,weights=new Float64Array(n);
      let max=-Infinity;
      for(let j=0;j<n;j++){
        let dot=0;
        for(let d=0;d<D;d++)dot+=q[offset+d]*mem.k[j][offset+d];
        weights[j]=dot/Math.sqrt(D);
        if(weights[j]>max)max=weights[j];
      }
      let sum=0;
      for(let j=0;j<n;j++){weights[j]=Math.exp(weights[j]-max);sum+=weights[j];}
      for(let d=0;d<D;d++){
        let v=0;
        for(let j=0;j<n;j++)v+=weights[j]/sum*mem.v[j][offset+d];
        attention[offset+d]=v;
      }
    }
    x=add(x,linear(attention,p.projW,p.projB,W));
    const ff=linear(norm(x,p.n2w,p.n2b),p.ff1W,p.ff1B,4*W);
    for(let i=0;i<ff.length;i++)ff[i]=gelu(ff[i]);
    x=add(x,linear(ff,p.ff2W,p.ff2B,W));
  }
  return linear(norm(x,m.normW,m.normB),m.head,null,V);
}
export function encodeMessages(messages,{maxInputTokens=192}={}){
  let tokens=[];
  for(const m of messages.slice(-5))tokens.push(m.role==='user'?USER:ASSISTANT,...Buffer.from(m.content,'utf8'),END);
  tokens.push(ASSISTANT);
  if(tokens.length>maxInputTokens){
    tokens=[];
    const m=messages.at(-1);
    tokens.push(USER,...Buffer.from(m.content,'utf8'),END,ASSISTANT);
  }
  if(tokens.length>maxInputTokens)throw Error('Message too long for the scratch model');
  return tokens;
}
// Nucleus sampling, temperature and repetition penalties replace greedy argmax.
// In this model a token is a UTF-8 byte, not an entire word. Preserve the
// existing 260-token checkpoint/worker protocol until the model is retrained.
export function sampleNextToken(logits,{
  recent=[],temperature=.76,topK=32,topP=.90,rng=Math.random,minOutput=0,step=0
}={}){
  if(!logits||logits.length!==V)throw Error('Invalid token distribution');
  if(!(temperature>0&&temperature<=2)||!(topP>0&&topP<=1))throw Error('Invalid sampling settings');
  const counts=new Map();
  for(const n of recent.slice(-96))counts.set(n,(counts.get(n)||0)+1);
  const choices=[];
  for(let id=0;id<V;id++){
    if(id===PAD||id===USER||id===ASSISTANT)continue;
    if(id===END && step<minOutput)continue;
    let value=logits[id];
    if(!Number.isFinite(value))continue;
    const repeat=Math.min(counts.get(id)||0,8);
    if(repeat){
      const p=1+.09*repeat;
      value=value>0?value/p:value*p;
    }
    // Prevent exact 3-byte loops while preserving ordinary doubled letters.
    if(recent.length>=2){
      const a=recent.at(-2),b=recent.at(-1);
      for(let j=2;j<recent.length;j++){
        if(recent[j-2]===a&&recent[j-1]===b&&recent[j]===id){
          value-=2.2;break;
        }
      }
    }
    choices.push({id,score:value/temperature});
  }
  choices.sort((a,b)=>b.score-a.score);
  const best=choices.slice(0,Math.max(1,Math.min(V,topK|0)));
  const max=best[0]?.score;
  if(!Number.isFinite(max))throw Error('No valid output tokens');
  let all=0;
  for(const item of best){item.weight=Math.exp(item.score-max);all+=item.weight;}
  const nucleus=[];let running=0;
  for(const item of best){
    nucleus.push(item);running+=item.weight;
    if(running/all>=topP)break;
  }
  const total=nucleus.reduce((v,item)=>v+item.weight,0);
  let pick=Math.max(0,Math.min(.999999999,Number(rng()))) * total;
  for(const item of nucleus){
    pick-=item.weight;
    if(pick<=0)return item.id;
  }
  return nucleus.at(-1).id;
}
export function scratchGenerate(checkpoint,messages,{
  maxNewTokens=64,temperature=.76,topP=.90,rng=Math.random
}={}){
  if(!Number.isInteger(maxNewTokens)||maxNewTokens<1||maxNewTokens>128)throw Error('Invalid generation limit');
  const model=Buffer.isBuffer(checkpoint)?parseCheckpoint(checkpoint):checkpoint;
  const cache=model.layers.map(()=>({k:[],v:[]})),tokens=encodeMessages(messages);
  let logits;
  for(const token of tokens)logits=logitsForToken(model,token,cache);
  const out=[];
  for(let t=0;t<maxNewTokens;t++){
    const next=sampleNextToken(logits,{
      recent:out,temperature,topP,rng,minOutput:5,step:t
    });
    if(next===END)break;
    out.push(next);
    logits=logitsForToken(model,next,cache);
  }
  try{return decoder.decode(Uint8Array.from(out)).trim();}catch{return '';}
}

// A bounded next-byte validation score. Used to reject obviously regressive
// training updates before they are merged into the community candidate.
export function scoreNextByte(modelOrBytes,messages,expectedByte){
  if(!Number.isInteger(expectedByte)||expectedByte<0||expectedByte>255)throw Error('Invalid target byte');
  const model=Buffer.isBuffer(modelOrBytes)?parseCheckpoint(modelOrBytes):modelOrBytes;
  const cache=model.layers.map(()=>({k:[],v:[]})),ids=encodeMessages(messages,{maxInputTokens:128});
  let logits;
  for(const token of ids)logits=logitsForToken(model,token,cache);
  let largest=-Infinity;
  for(let i=0;i<logits.length;i++)if(Number.isFinite(logits[i])&&logits[i]>largest)largest=logits[i];
  if(!Number.isFinite(largest))throw Error('Invalid model predictions');
  let denominator=0;
  for(let i=0;i<logits.length;i++){
    if(!Number.isFinite(logits[i]))throw Error('Invalid model predictions');
    denominator+=Math.exp(logits[i]-largest);
  }
  const loss=largest+Math.log(denominator)-logits[expectedByte];
  if(!Number.isFinite(loss)||loss<0||loss>1000)throw Error('Invalid validation loss');
  return loss;
}
