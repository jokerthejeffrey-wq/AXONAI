import crypto from 'node:crypto';
export const CHUNK_BYTES=2_500_000; // below Vercel's 4.5 MB request limit
export const MAX_CHECKPOINT=32*1024*1024;
export const BATCH=8;
export const MANIFEST={protocol:2,architecture:{vocab_size:260,width:256,layers:4,heads:8,context:512},tokenizer:'utf8-bytes-v1',max_uses:1};
export function lesson(n) {
  if(n===0)return {concept:'greeting',messages:[{role:'user',content:'hello'}],response:'Hello!'};
  const op=(n-1)%9, k=Math.floor((n-1)/9)+2;
  const cases=[
    ['addition',`What is ${k} + ${k+3}?`,String(2*k+3)],
    ['subtraction',`Calculate ${k+20} minus ${k}.`,'20'],
    ['multiplication',`What is ${k} times 7?`,String(k*7)],
    ['parity',`Is ${k} an even or odd number?`,k%2?'odd':'even'],
    ['comparison',`Which is larger, ${k+1} or ${k+2}?`,String(k+2)],
    ['division',`What is ${k*3} divided by 3?`,String(k)],
    ['unit conversion',`How many centimeters are in ${k} meters?`,String(k*100)],
    ['square',`What is ${k} squared?`,String(k*k)],
    ['binary',`Write ${k} in binary.`,k.toString(2)],
  ];
  const [concept,content,response]=cases[op];
  return {concept,messages:[{role:'user',content}],response};
}
export function makeLessons(start){return Array.from({length:BATCH},(_,i)=>lesson(start+i));}
export function sha256(buffer){return crypto.createHash('sha256').update(buffer).digest('hex');}
export function idOK(id){return typeof id==='string' && /^[a-f0-9]{32}$/.test(id);}
export function workerOK(id){return typeof id==='string' && /^[a-z0-9-]{10,80}$/i.test(id);}
export function tokenOK(tok){return typeof tok==='string' && /^[a-f0-9]{64}$/.test(tok);}
export function packChunks(buf,size=CHUNK_BYTES){const result=[];for(let i=0;i<buf.length;i+=size)result.push(buf.subarray(i,i+size));return result;}
// Safetensors F32 merger. No untrusted executable code, pickle or dynamic imports.
export function mergeWeights(base, current, candidate) {
  function describe(buf){
    if(!Buffer.isBuffer(buf)||buf.length<16||buf.length>MAX_CHECKPOINT)throw Error('Invalid checkpoint length');
    const n=Number(buf.readBigUInt64LE(0));
    if(!Number.isSafeInteger(n)||n<10||n>100_000||n+8>buf.length)throw Error('Invalid tensor header');
    let data;try{data=JSON.parse(buf.subarray(8,8+n).toString('utf8'));}catch{throw Error('Invalid checkpoint JSON')}
    const keys=Object.keys(data).filter(k=>k!=='__metadata__').sort();
    if(!keys.length||keys.length>300)throw Error('Invalid tensors');
    let entries=[];
    for(const key of keys){
      const t=data[key];
      if(!t||t.dtype!=='F32'||!Array.isArray(t.shape)||t.shape.length>5||t.shape.some(x=>!Number.isSafeInteger(x)||x<=0))throw Error('Invalid tensor dtype/shape');
      if(!Array.isArray(t.data_offsets)||t.data_offsets.length!==2)throw Error('Invalid tensor offsets');
      const [s,e]=t.data_offsets;const elems=t.shape.reduce((a,b)=>a*b,1);
      if(!Number.isSafeInteger(elems)||!Number.isSafeInteger(s)||!Number.isSafeInteger(e)||s<0||e-s!==elems*4||e+n+8>buf.length)throw Error('Invalid tensor size');
      entries.push({key,shape:t.shape.join(','),s,e});
    }
    return {offset:n+8,entries};
  }
  const b=describe(base),c=describe(current),p=describe(candidate);
  if(b.entries.length!==c.entries.length||b.entries.length!==p.entries.length)throw Error('Model architecture mismatch');
  const out=Buffer.from(current);
  for(let i=0;i<b.entries.length;i++){
    const eb=b.entries[i],ec=c.entries[i],ep=p.entries[i];
    if(eb.key!==ec.key||eb.key!==ep.key||eb.shape!==ec.shape||eb.shape!==ep.shape)throw Error('Tensor shape mismatch');
    const steps=(eb.e-eb.s)/4;
    for(let k=0;k<steps;k++){
      const vBase=base.readFloatLE(b.offset+eb.s+k*4);
      const vNow=current.readFloatLE(c.offset+ec.s+k*4);
      const vCand=candidate.readFloatLE(p.offset+ep.s+k*4);
      const delta=vCand-vBase;
      if(!Number.isFinite(vBase)||!Number.isFinite(vNow)||!Number.isFinite(vCand)||Math.abs(delta)>0.05001)throw Error('Invalid or excessive model weight update');
      const result=vNow+0.5*delta;
      if(!Number.isFinite(result))throw Error('Invalid merged weights');
      out.writeFloatLE(result,c.offset+ec.s+k*4);
    }
  }
  return out;
}
