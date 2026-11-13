import {identityAnswer,validateMessages} from '../server/config.js';
import {originAllowed,parseChatBody,rateLimit,turnstileConfig,verifyTurnstile} from '../server/security.js';
import {simpleArithmetic} from '../server/arithmetic.js';
import {getQuality,inferFromStudent} from '../server/coordinator.js';

function rejectRate(res,seconds,message){
  res.setHeader('Retry-After',String(seconds));
  return res.status(429).json({error:message});
}
export default async function handler(req,res){
  res.setHeader('Cache-Control','no-store');
  if(req.method!=='POST'){res.setHeader('Allow','POST');return res.status(405).json({error:'Use POST.'});}
  if(!originAllowed(req))return res.status(403).json({error:'This request must come from AXON AI.'});
  const wait=rateLimit(req,'requests');if(wait)return rejectRate(res,wait,'Too many requests. Try again shortly.');
  let body,messages;
  try{body=parseChatBody(req);messages=validateMessages(body.messages);}
  catch(error){return res.status(error.status||400).json({error:error.status?error.message:'Invalid message.'});}
  const verification=await verifyTurnstile(req,body.turnstileToken,turnstileConfig());
  if(!verification.ok)return res.status(verification.status).json({error:verification.error});
  const limited=rateLimit(req,'generations');
  if(limited)return rejectRate(res,limited,'Message limit reached. Try again shortly.');
  const last=messages.at(-1).content;
  const identity=identityAnswer(last);
  if(identity)return res.status(200).json({reply:identity,mode:'local-basic',model:'AXON AI'});
  const arithmetic=simpleArithmetic(last);
  if(arithmetic!==null)return res.status(200).json({reply:arithmetic,mode:'local-calculation',model:'AXON AI'});
  try{
    const quality=await getQuality();
    const result=await inferFromStudent(messages,quality);
    return res.status(200).json({reply:result.reply,mode:result.mode,model:'AXON AI',version:result.version});
  }catch(error){
    console.error('AXON local chat:',String(error?.name||'Error').slice(0,40));
    return res.status(503).json({error:'Local model is temporarily unavailable.',code:'student_unavailable'});
  }
}
