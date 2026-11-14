// The chat only calls AXON AI's own scratch-trained Transformer. No browser weights.
let messages=[],busy=false;
const prompt=document.getElementById('prompt');
const send=document.getElementById('send');
const statusEl=document.getElementById('status');
const box=document.getElementById('messages');
const previewEl=document.getElementById('experimental-preview');
function bubble(role,text){
  const div=document.createElement('div');div.className='message '+role;div.textContent=text;
  box.append(div);window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});return div;
}
function resize(){prompt.style.height='auto';prompt.style.height=Math.min(prompt.scrollHeight,130)+'px';}
prompt.oninput=resize;
function showChatMode(mode){
  const label=mode==='student'?'· scratch-trained student':mode==='experimental-student'?'· unverified scratch student':'· student training';
  document.getElementById('chat-mode').textContent=label;
}
async function status(){
  try{
    const r=await axonFetch('/api/status');if(!r.ok)throw Error('Status unavailable');
    const s=await r.json();showChatMode(s.chat_mode==='scratch-student'?'student':'learning');
    previewEl.disabled=!s.student_preview_available;
    if(!s.student_preview_available)previewEl.checked=false;
    if(s.chat_student_ready){statusEl.textContent='AXON AI is running its own trained student.';}
    else if(s.student_preview_available){statusEl.textContent='Student checkpoint v'+s.training_version+' is unverified. Enable experimental preview to inspect it.';}
    else{statusEl.textContent='The AXON student is training. Chat will unlock as approved checkpoints and evaluations become available.';}
  }catch{showChatMode('learning');statusEl.textContent='Training status is unavailable.';}
}
document.getElementById('new').onclick=()=>{
  if(busy)return;messages=[];box.replaceChildren();
  document.getElementById('welcome').hidden=false;prompt.value='';resize();status();prompt.focus();
};
document.querySelectorAll('.suggestions button').forEach(b=>b.onclick=()=>{
  prompt.value=b.textContent;document.getElementById('composer').requestSubmit();
});
prompt.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();document.getElementById('composer').requestSubmit();}
});
async function generate(outgoing,token,bubbleEl){
  const response=await axonFetch('/api/chat',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({messages:outgoing,stream:true,preview:previewEl.checked,...(token?{turnstileToken:token}:{})})
  });
  if(!response.ok){
    const data=await response.json();
    throw Error(data.error||'Student inference unavailable.');
  }
  const type=response.headers.get('content-type')||'';
  if(!type.includes('application/x-ndjson'))throw Error('Unexpected response format.');
  const reader=response.body.getReader(),decoder=new TextDecoder();
  let pending='',reply='',done=false,mode='learning';
  while(true){
    const item=await reader.read();
    pending+=decoder.decode(item.value||new Uint8Array(),{stream:!item.done});
    let pos;
    while((pos=pending.indexOf('\n'))>=0){
      const line=pending.slice(0,pos);pending=pending.slice(pos+1);
      if(!line)continue;
      const event=JSON.parse(line);
      if(event.type==='error')throw Error(event.error||'Student stream failed.');
      if(event.type==='meta'){mode=event.mode;showChatMode(mode);}
      if(event.type==='delta'){reply+=event.text;bubbleEl.textContent=reply;}
      if(event.type==='done')done=true;
    }
    if(item.done)break;
  }
  if(!done||!reply.trim())throw Error('AXON student has not learned to output a reply yet.');
  return {reply:reply.trim(),mode};
}
document.getElementById('composer').onsubmit=async e=>{
  e.preventDefault();const text=prompt.value.trim();if(!text||busy)return;
  busy=true;send.disabled=true;document.getElementById('new').disabled=true;
  let userBubble=null,assistantBubble=null;
  try{
    await window.axonSecurityReady;
    const token=window.axonSecurityToken();
    document.getElementById('welcome').hidden=true;
    userBubble=bubble('user',text);prompt.value='';resize();
    statusEl.textContent=previewEl.checked?'Sampling unverified AXON weights…':'Sampling AXON student tokens…';
    const outgoing=[...messages.slice(-12),{role:'user',content:text}];
    assistantBubble=bubble('assistant','');
    const result=await generate(outgoing,token,assistantBubble);
    assistantBubble.textContent=result.reply;
    messages=[...outgoing,{role:'assistant',content:result.reply}];
    statusEl.textContent=result.mode==='experimental-student'?'Experimental output: not verified for accuracy or coherence.':'';
  }catch(error){
    userBubble?.remove();assistantBubble?.remove();
    if(!messages.length)document.getElementById('welcome').hidden=false;
    prompt.value=text;resize();
    statusEl.textContent=error.message||'AXON student unavailable.';
  }finally{
    window.axonSecurityReset?.();
    busy=false;send.disabled=false;document.getElementById('new').disabled=false;prompt.focus();
  }
};
status();
