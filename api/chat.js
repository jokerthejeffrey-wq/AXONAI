// AXON AI public chat. The ONLY generator is the community-trained checkpoint.
// No pretrained provider, browser model download, or fixed conversational answer.
import {validateMessages} from '../server/config.js';
import {originAllowed,parseChatBody,rateLimit,turnstileConfig,verifyTurnstile} from '../server/security.js';
import {getQuality} from '../server/coordinator.js';
import {loadStudent,readableReply} from '../server/scratch-chat.js';
import {scratchGenerate,scratchStream} from '../server/scratch-infer.js';

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

  const quality=await getQuality();
  const version=Number(quality?.version||0);
  const ready=version>0 && quality?.tested===true && quality?.ready===true;
  const preview=body.preview===true;
  if(!Number.isSafeInteger(version)||version<1||(!ready&&!preview)){
    return res.status(503).json({
      code:'student_training',
      error:version<1
        ?'AXON AI has not received an approved student checkpoint yet. No pretrained model will answer instead.'
        :'AXON AI is still training and has not passed chat evaluation. Enable experimental preview to inspect its own unverified output.',
      version:Number.isSafeInteger(version)?version:0,mode:'learning'
    });
  }
  let model;
  try{model=await loadStudent(version);}
  catch(error){
    console.error('AXON student loading:',String(error?.name||'Error').slice(0,40));
    return res.status(503).json({code:'checkpoint_unavailable',error:'AXON student checkpoint is temporarily unavailable.'});
  }
  const mode=ready?'student':'experimental-student';
  if(body.stream!==true){
    try{
      const reply=scratchGenerate(model,messages,{maxNewTokens:128,temperature:.55,topK:16,topP:.8,repetitionPenalty:1.15});
      if(!reply || (ready&&!readableReply(reply)))
        return res.status(503).json({code:'student_training',error:'AXON student could not produce a readable reply yet.'});
      return res.status(200).json({reply,mode,version,tokenizer:'utf8-bytes-v1',verified:ready});
    }catch(error){return res.status(503).json({code:'generation_unavailable',error:'AXON student generation failed.'});}
  }
  res.setHeader('Content-Type','application/x-ndjson; charset=utf-8');
  res.setHeader('X-Content-Type-Options','nosniff');
  res.flushHeaders?.();
  res.write(JSON.stringify({type:'meta',mode,model:'AXON AI scratch Transformer',version,verified:ready})+'\n');
  try{
    let output='';
    for await(const chunk of scratchStream(model,messages,{maxNewTokens:128,temperature:.55,topK:16,topP:.8,repetitionPenalty:1.15})){
      if(res.destroyed)break;
      output+=chunk;
      if(output.length>1024)throw Error('Model output limit');
      res.write(JSON.stringify({type:'delta',text:chunk})+'\n');
    }
    if(!res.destroyed){
      if(!output.trim())res.write(JSON.stringify({type:'error',error:'Student did not produce text yet.'})+'\n');
      res.end(JSON.stringify({type:'done'})+'\n');
    }
  }catch(error){
    console.warn('AXON scratch stream interrupted:',String(error?.name||'Error').slice(0,40));
    if(!res.destroyed)res.end(JSON.stringify({type:'error',error:'Student generation was interrupted.'})+'\n');
  }
}
