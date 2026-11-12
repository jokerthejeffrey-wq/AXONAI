// Public coordinator status, protected coordinator inference via server-side bearer token.
function coordinatorBase(){
  const raw=process.env.AXON_COORDINATOR_URL?.trim();
  if(!raw)return null;
  const url=new URL(raw);
  if(url.protocol!=='https:' && !(url.protocol==='http:' && ['127.0.0.1','localhost'].includes(url.hostname)))return null;
  if(url.username||url.password||url.search||url.hash)return null;
  return url.origin+url.pathname.replace(/\/$/,'');
}
async function query(path,init={},timeoutMs=5000){
  const base=coordinatorBase();if(!base) return null;
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),timeoutMs);
  try{const r=await fetch(base+path,{...init,signal:controller.signal,cache:'no-store'});
   if(!r.ok)throw Error('Coordinator unavailable');return await r.json();
  }finally{clearTimeout(timeout)}
}
export async function getQuality(){
  try{return await query('/api/quality')}catch{return null}
}
export async function inferFromStudent(messages,quality){
  if(quality?.ready!==true||!Number.isSafeInteger(quality?.version)||!process.env.AXON_INFERENCE_TOKEN)return null;
  try{const data=await query('/api/infer',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+process.env.AXON_INFERENCE_TOKEN},body:JSON.stringify({messages,version:quality.version})},30000);
   if(typeof data?.reply!=='string'||!data.reply.trim())return null;return data.reply.trim();
  }catch{return null}
}
