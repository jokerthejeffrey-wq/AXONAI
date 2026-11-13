import { identityAnswer, validateMessages } from '../server/config.js';
import { originAllowed, parseChatBody, rateLimit, turnstileConfig, verifyTurnstile } from '../server/security.js';
import { simpleArithmetic } from '../server/arithmetic.js';
import { getQuality, inferFromStudent } from '../server/coordinator.js';
import { gatewayGenerate, GatewayError } from '../server/gateway.js';

function rejectRate(res, seconds, message) {
  res.setHeader('Retry-After', String(seconds));
  return res.status(429).json({ error: message });
}

export default async function handler(req,res) {
  res.setHeader('Cache-Control','no-store');
  if (req.method !== 'POST') {
    res.setHeader('Allow','POST');
    return res.status(405).json({error:'Use POST.'});
  }
  if (!originAllowed(req)) return res.status(403).json({error:'This request must come from AXON AI.'});
  const reqWait=rateLimit(req,'requests');
  if (reqWait) return rejectRate(res,reqWait,'Too many requests. Try again shortly.');
  let body,messages;
  try {
    body=parseChatBody(req);
    messages=validateMessages(body.messages);
  } catch (error) {
    return res.status(error.status||400).json({error:error.status ? error.message : 'Invalid message.'});
  }
  const verification=await verifyTurnstile(req,body.turnstileToken,turnstileConfig());
  if (!verification.ok) return res.status(verification.status).json({error:verification.error});
  const genWait=rateLimit(req,'generations');
  if (genWait) return rejectRate(res,genWait,'Message limit reached. Try again shortly.');

  const last=messages.at(-1).content;
  const identity=identityAnswer(last);
  if (identity) return res.status(200).json({reply:identity,mode:'identity',model:'AXON AI'});
  const arithmetic=simpleArithmetic(last);
  if (arithmetic !== null) return res.status(200).json({reply:arithmetic,mode:'calculation',model:'AXON AI'});

  try {
    // Use the independently trained student only after the cloud coordinator
    // reports an approved, evaluated checkpoint and successfully serves it.
    const quality=await getQuality();
    const studentReply=await inferFromStudent(messages,quality);
    if (studentReply) return res.status(200).json({reply:studentReply,mode:'student',model:'AXON AI',version:quality.version});

    // The pretrained reply is generated remotely. This Vercel Function does
    // not need a local AI model, a GPU, or a RAM-intensive model download.
    const reply=await gatewayGenerate(messages,req);
    return res.status(200).json({reply,mode:'pretrained',model:'AXON AI'});
  } catch (error) {
    const code=error instanceof GatewayError ? error.code : 'unknown';
    // Only log a status category. Never include private tokens or raw prompts.
    console.error('AXON AI chat backend:',code);
    if (code==='auth_missing' || code==='auth_rejected' || code==='access_denied') return res.status(503).json({
      error:'Chat is temporarily unavailable. Please try again later.',code
    });
    if (code==='verification_required') return res.status(503).json({
      error:'Chat is temporarily unavailable. Please try again later.',code
    });
    if (code==='billing') return res.status(503).json({
      error:'Chat credits are unavailable. The site owner must check AI Gateway billing.',code
    });
    if (code==='model_unavailable' || code==='config') return res.status(503).json({
      error:'Chat model is not configured correctly.',code
    });
    if (code==='rate_limited') return res.status(503).json({error:'Chat is temporarily busy. Try again shortly.',code});
    if (code==='timeout') return res.status(504).json({error:'Chat timed out. Please try again.',code});
    return res.status(503).json({error:'Chat is temporarily unavailable. Please try again.',code});
  }
}
