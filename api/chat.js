import {validateMessages} from '../server/config.js';
import {originAllowed,parseChatBody,rateLimit,turnstileConfig,verifyTurnstile} from '../server/security.js';
import {beginGateway,gatewayDeltas} from '../server/gateway-stream.js';

export default async function handler(req,res){
  res.setHeader('Cache-Control','no-store');
  if(req.method!=='POST'){res.setHeader('Allow','POST');return res.status(405).json({error:'Use POST.'});}
  if(!originAllowed(req))return res.status(403).json({error:'This request must come from AXON AI.'});
  const wait=rateLimit(req,'requests');
  if(wait){res.setHeader('Retry-After',String(wait));return res.status(429).json({error:'Too many requests. Try again shortly.'});}
  let body,messages;
  try{body=parseChatBody(req);messages=validateMessages(body.messages);}
  catch(error){return res.status(error.status||400).json({error:error.status?error.message:'Invalid message.'});}
  const verification=await verifyTurnstile(req,body.turnstileToken,turnstileConfig());
  if(!verification.ok)return res.status(verification.status).json({error:verification.error});
  const limited=rateLimit(req,'generations');
  if(limited){res.setHeader('Retry-After',String(limited));return res.status(429).json({error:'Message limit reached. Try again shortly.'});}

  const stream=body.stream===true;
  const gateway=await beginGateway(messages,{stream});
  if(!gateway){
    // The browser uses real on-device pretrained weights when cloud access is blocked.
    // Never return an untrained student's byte samples or a fixed greeting.
    return res.status(503).json({code:'browser_model_required',error:'Cloud inference is unavailable; use the on-device pretrained model.'});
  }
  if(!stream){
    try{
      const data=await gateway.json(),reply=data?.choices?.[0]?.message?.content;
      if(typeof reply!=='string'||!reply.trim())throw Error('Empty model reply');
      return res.status(200).json({reply:reply.trim(),mode:'assisted',model:'Qwen3 14B (pretrained)'});
    }catch{return res.status(502).json({code:'browser_model_required',error:'Cloud model reply failed.'});}
  }
  res.setHeader('Content-Type','application/x-ndjson; charset=utf-8');
  res.setHeader('X-Content-Type-Options','nosniff');
  res.flushHeaders?.();
  res.write(JSON.stringify({type:'meta',mode:'assisted',model:'Qwen3 14B (pretrained)'})+'\n');
  try{
    let length=0;
    for await(const part of gatewayDeltas(gateway)){
      if(res.destroyed)break;
      length+=part.length;
      if(length>8000)throw Error('Reply too long');
      res.write(JSON.stringify({type:'delta',text:part})+'\n');
    }
    if(!res.destroyed)res.end(JSON.stringify({type:'done'})+'\n');
  }catch(error){
    console.warn('AXON cloud stream failed:',String(error?.name||'Error').slice(0,40));
    if(!res.destroyed)res.end(JSON.stringify({type:'error',error:'Cloud generation interrupted. Try again.'})+'\n');
  }
}
