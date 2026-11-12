// Hosted inference: no large model downloads in ephemeral Vercel functions.
// The token is only used server-side and is never returned to browsers.
export class GatewayError extends Error {
  constructor(message, status=503) { super(message); this.name='GatewayError'; this.status=status; }
}

export function gatewayToken(req,env=process.env){
  const staticKey = env.AI_GATEWAY_API_KEY?.trim();
  if(staticKey) return staticKey;
  const oidc = env.VERCEL_OIDC_TOKEN?.trim();
  if(oidc) return oidc;
  // Vercel's runtime can inject OIDC on the server-side Request headers.
  if(env.VERCEL){
    const raw=req?.headers?.['x-vercel-oidc-token'];
    if(typeof raw==='string'&&raw.length>20)return raw.trim();
  }
  return null;
}

export async function gatewayGenerate(messages, req, {fetcher=fetch,env=process.env}={}){
  const token=gatewayToken(req,env);
  if(!token) throw new GatewayError('Vercel AI Gateway authentication is not configured.',503);
  const chosen=env.AXON_GATEWAY_MODEL||'openai/gpt-4.1-mini';
  // Allow the owner to choose another model via an environment variable.
  // Never accept a model ID from an untrusted browser request.
  if(!/^[a-z0-9-]+\/[a-zA-Z0-9._:-]+$/.test(chosen))throw new GatewayError('Invalid model configuration.');
  let r;
  try{
    r=await fetcher('https://ai-gateway.vercel.sh/v1/chat/completions', {
      method:'POST',
      headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'},
      body:JSON.stringify({
        model:chosen,
        messages:[{role:'system',content:'You are AXON AI, a community AI assistant. Answer clearly, accurately, and concisely. You are pretrained assistance while a separate community-built student model is in training. Do not pretend you were trained from scratch by AXON AI. If unsure, say so.'},...messages.slice(-11)],
        max_tokens:450,temperature:0.35,stream:false,
      }),
      signal:AbortSignal.timeout(45000)
    });
  }catch(error){
    throw new GatewayError(error.name==='TimeoutError'?'The chat service timed out.':'The chat service could not be reached.');
  }
  if(!r.ok){
    const status=r.status;
    // Do not expose backend error bodies or authentication tokens to visitors.
    throw new GatewayError('AI Gateway returned HTTP '+status, status);
  }
  let data;
  try{data=await r.json();}catch{throw new GatewayError('Invalid response from AI Gateway.');}
  const content=data?.choices?.[0]?.message?.content;
  const answer=typeof content==='string'?content:Array.isArray(content)?content.filter(p=>p?.type==='text').map(p=>p.text||'').join('\n'):'';
  if(!answer?.trim())throw new GatewayError('AI Gateway returned an empty reply.');
  return answer.trim();
}
