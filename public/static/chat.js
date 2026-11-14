// The chat UI renders only model output, never preset greetings.
let messages=[],busy=false,modelWorker=null,workerSequence=0;
const prompt=document.getElementById('prompt');
const send=document.getElementById('send');
const statusEl=document.getElementById('status');
const box=document.getElementById('messages');

function bubble(role,text){
  const div=document.createElement('div');
  div.className='message '+role;
  div.textContent=text;
  box.append(div);
  window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});
  return div;
}
function resize(){
  prompt.style.height='auto';
  prompt.style.height=Math.min(prompt.scrollHeight,130)+'px';
}
prompt.oninput=resize;
document.getElementById('new').onclick=()=>{
  if(busy)return;
  messages=[];box.replaceChildren();
  document.getElementById('welcome').hidden=false;
  statusEl.textContent='';
  prompt.value='';resize();prompt.focus();
  showChatMode('');
};
document.querySelectorAll('.suggestions button').forEach(b=>b.onclick=()=>{
  prompt.value=b.textContent;
  document.getElementById('composer').requestSubmit();
});
prompt.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){
    e.preventDefault();
    document.getElementById('composer').requestSubmit();
  }
});
function showChatMode(mode,model=''){
  document.getElementById('chat-mode').textContent=
    mode==='assisted'?'· Qwen3 pretrained (cloud)':
    mode==='local-pretrained'?'· '+(model||'pretrained on-device'):'';
}
function getModelWorker(){
  if(!modelWorker)modelWorker=new Worker('/static/local-model-worker.js',{type:'module'});
  return modelWorker;
}
function generateInBrowser(outgoing,assistantBubble){
  const worker=getModelWorker(),id=++workerSequence;
  return new Promise((resolve,reject)=>{
    let done=false,reply='';
    const timer=setTimeout(()=>{
      cleanup();reject(Error('Model loading or generation took too long. Check your connection and available device memory.'));
    },300000);
    function cleanup(){
      clearTimeout(timer);
      worker.removeEventListener('message',onMessage);
      worker.removeEventListener('error',onError);
    }
    function onError(){
      cleanup();
      if(modelWorker===worker){modelWorker.terminate();modelWorker=null;}
      reject(Error('Browser model failed to start. Try a supported browser with more available memory.'));
    }
    function onMessage(event){
      const e=event.data||{};
      if(e.type==='status'){statusEl.textContent=e.text;return;}
      if(e.type==='meta'){showChatMode(e.mode,e.model);return;}
      if(e.type==='delta'){
        reply+=e.text;
        assistantBubble.textContent=reply;
        return;
      }
      if(e.type==='error'){cleanup();reject(Error(e.error||'Local model failed.'));return;}
      if(e.type==='done'){
        done=true;cleanup();
        resolve({reply:e.reply||reply,mode:'local-pretrained',model:e.model});
      }
    }
    worker.addEventListener('message',onMessage);
    worker.addEventListener('error',onError);
    statusEl.textContent='Loading real pretrained weights (first use may download hundreds of MB)…';
    worker.postMessage({type:'generate',id,messages:outgoing});
  });
}
async function generateFromCloud(outgoing,token,assistantBubble){
  const response=await axonFetch('/api/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({messages:outgoing,stream:true,...(token?{turnstileToken:token}:{})})
  });
  const type=response.headers.get('content-type')||'';
  if(!response.ok){
    const data=await response.json();
    if(data.code==='browser_model_required')return null;
    throw Error(data.error||'Cloud generation failed.');
  }
  if(!type.includes('application/x-ndjson')){
    const data=await response.json();
    if(!data?.reply)throw Error('Empty cloud model answer.');
    assistantBubble.textContent=data.reply;
    return data;
  }
  const reader=response.body.getReader(),decoder=new TextDecoder();
  let pending='',reply='',done=false,mode='assisted',model='';
  while(true){
    const item=await reader.read();
    pending+=decoder.decode(item.value||new Uint8Array(),{stream:!item.done});
    let pos;
    while((pos=pending.indexOf('\n'))>=0){
      const line=pending.slice(0,pos);
      pending=pending.slice(pos+1);
      if(!line)continue;
      const event=JSON.parse(line);
      if(event.type==='error')throw Error(event.error||'Cloud model stream failed.');
      if(event.type==='meta'){mode=event.mode;model=event.model;showChatMode(mode,model);}
      if(event.type==='delta'){reply+=event.text;assistantBubble.textContent=reply;}
      if(event.type==='done')done=true;
    }
    if(item.done)break;
  }
  if(!done||!reply.trim())throw Error('Cloud model did not complete a reply.');
  return {reply:reply.trim(),mode,model};
}
document.getElementById('composer').onsubmit=async e=>{
  e.preventDefault();
  const text=prompt.value.trim();
  if(!text||busy)return;
  busy=true;send.disabled=true;document.getElementById('new').disabled=true;
  let userBubble=null,assistantBubble=null;
  try{
    await window.axonSecurityReady;
    const token=window.axonSecurityToken();
    document.getElementById('welcome').hidden=true;
    userBubble=bubble('user',text);
    prompt.value='';resize();
    statusEl.textContent='Starting pretrained generation…';
    const outgoing=[...messages.slice(-12),{role:'user',content:text}];
    assistantBubble=bubble('assistant','');
    let result=await generateFromCloud(outgoing,token,assistantBubble);
    if(!result)result=await generateInBrowser(outgoing,assistantBubble);
    if(!result?.reply?.trim())throw Error('The model produced no reply.');
    assistantBubble.textContent=result.reply;
    showChatMode(result.mode,result.model);
    messages=[...outgoing,{role:'assistant',content:result.reply}];
    statusEl.textContent='';
  }catch(error){
    userBubble?.remove();assistantBubble?.remove();
    if(!messages.length)document.getElementById('welcome').hidden=false;
    prompt.value=text;resize();
    statusEl.textContent=error.message||'The language model is unavailable.';
  }finally{
    window.axonSecurityReset?.();
    busy=false;send.disabled=false;document.getElementById('new').disabled=false;prompt.focus();
  }
};
