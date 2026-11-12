// Optional stronger hosted chat. Never send the API key to a browser.
export function hasHosted(){return Boolean(process.env.AXON_OPENROUTER_KEY?.trim());}
export async function hostedGenerate(messages){
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),35000);
  try {
    const response=await fetch('https://openrouter.ai/api/v1/chat/completions',{
      method:'POST',signal:controller.signal,headers:{
        'Authorization':'Bearer '+process.env.AXON_OPENROUTER_KEY,
        'Content-Type':'application/json','HTTP-Referer':'https://vercel.app','X-Title':'AXON AI'
      },body:JSON.stringify({model:process.env.AXON_HOSTED_MODEL||'qwen/qwen3-8b',
       messages:[{role:'system',content:'You are AXON AI. Answer carefully, accurately and concisely. Do not invent facts. You are a pretrained helper while the AXON AI student model is developed.'},...messages.slice(-11)],
       max_tokens:350,temperature:0.3})
    });
    if(!response.ok)throw new Error('Hosted model HTTP '+response.status);
    const data=await response.json();
    const text=data?.choices?.[0]?.message?.content;
    if(typeof text!=='string'||!text.trim())throw new Error('Hosted model returned an empty reply');
    return text.trim();
  } finally {clearTimeout(timeout);}
}
