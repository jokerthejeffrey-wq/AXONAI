// AXON AI optionally uses a real pretrained cloud model. Never fabricate output.
const URL='https://ai-gateway.vercel.sh/v1/chat/completions';
let coolingUntil=0;
export async function beginGateway(messages,{stream=false,fetcher=fetch,now=Date.now}={}){
  const key=process.env.AI_GATEWAY_API_KEY?.trim();
  if(!key||now()<coolingUntil)return null;
  try{
    const response=await fetcher(URL,{
      method:'POST',
      headers:{'Authorization':'Bearer '+key,'Content-Type':'application/json'},
      body:JSON.stringify({
        model:'alibaba/qwen-3-14b',
        messages:[
          {role:'system',content:'You are the pretrained chat assistant for AXON AI. Reply naturally and accurately to the user. You are a separate pretrained model while AXON trains its own student; never claim your words came from the untrained student.'},
          ...messages.slice(-9)
        ],
        max_tokens:256,
        stream
      }),
      signal:AbortSignal.timeout(25000)
    });
    if(!response.ok){
      if([401,402,403,429].includes(response.status))coolingUntil=now()+5*60*1000;
      console.warn('AXON cloud chat unavailable:',response.status);
      return null;
    }
    return response;
  }catch(error){
    console.warn('AXON cloud chat connection failed:',String(error?.name||'Error').slice(0,40));
    return null;
  }
}

// AI Gateway streams OpenAI-compatible SSE. Convert *actual model tokens*
// to the site's existing newline-delimited JSON streaming protocol.
export async function* gatewayDeltas(response){
  if(!response.body?.getReader)throw Error('Gateway stream not readable');
  const reader=response.body.getReader(),decoder=new TextDecoder();
  let pending='',done=false;
  try{
    while(true){
      const chunk=await reader.read();
      pending+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});
      let pos;
      while((pos=pending.indexOf('\n'))>=0){
        const line=pending.slice(0,pos).trim();
        pending=pending.slice(pos+1);
        if(!line.startsWith('data:'))continue;
        const raw=line.slice(5).trim();
        if(raw==='[DONE]'){done=true;return;}
        if(!raw)continue;
        const event=JSON.parse(raw);
        if(event.error)throw Error('Gateway generation failed');
        const content=event.choices?.[0]?.delta?.content;
        if(typeof content==='string'&&content)yield content;
      }
      if(chunk.done)break;
    }
    if(!done)throw Error('Gateway stream ended unexpectedly');
  }finally{reader.releaseLock();}
}
