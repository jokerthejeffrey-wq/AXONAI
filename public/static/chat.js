let messages = [], busy = false;
const prompt = document.getElementById('prompt');
const send = document.getElementById('send');
const statusEl = document.getElementById('status');
const box = document.getElementById('messages');

function bubble(role, text) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = text; // Never interpret chat messages or model output as HTML.
  box.append(div);
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  return div;
}
function resize() {
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 130) + 'px';
}
prompt.oninput = resize;
document.getElementById('new').onclick = () => {
  if (busy) return;
  messages = [];
  box.replaceChildren();
  document.getElementById('welcome').hidden = false;
  statusEl.textContent = '';
  prompt.value = '';
  resize();
  prompt.focus();
};
document.querySelectorAll('.suggestions button').forEach(b => b.onclick = () => {
  prompt.value = b.textContent;
  document.getElementById('composer').requestSubmit();
});
prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    document.getElementById('composer').requestSubmit();
  }
});

document.getElementById('composer').onsubmit = async e => {
  e.preventDefault();
  const text = prompt.value.trim();
  if (!text || busy) return;
  busy = true;
  send.disabled = true;
  document.getElementById('new').disabled = true;
  let userBubble = null, assistantBubble = null;
  try {
    await window.axonSecurityReady;
    const token = window.axonSecurityToken();
    document.getElementById('welcome').hidden = true;
    userBubble = bubble('user', text);
    prompt.value = '';
    resize();
    statusEl.textContent = 'Generating answer..';
    const outgoing = [...messages.slice(-12), { role: 'user', content: text }];
    let data;
    for (let attempt = 0; attempt < 30; attempt++) {
      // A pending challenge can expire during a long capacity wait.
      if (token && !window.axonSecurityToken()) throw Error('Complete security verification again.');
      const response = await axonFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: outgoing, stream: true, ...(token ? { turnstileToken: token } : {}) })
      });
      if(response.ok && (response.headers.get('content-type')||'').includes('application/x-ndjson')){
        assistantBubble=bubble('assistant','');
        const reader=response.body.getReader(),decoder=new TextDecoder();let pending='',reply='',done=false;
        while(true){
          const part=await reader.read();pending+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});
          let newline;
          while((newline=pending.indexOf('\n'))>=0){
            const line=pending.slice(0,newline);pending=pending.slice(newline+1);if(!line)continue;
            const event=JSON.parse(line);
            if(event.type==='error')throw Error(event.error);
            if(event.type==='meta')statusEl.textContent='Experimental student · checkpoint '+event.version;
            if(event.type==='delta'){reply+=event.text;assistantBubble.textContent=reply;}
            if(event.type==='done')done=true;
          }
          if(part.done)break;
        }
        if(!done||!reply.trim())throw Error('The student did not finish a usable response. It still needs training.');
        data={reply};break;
      }
      data = await response.json();
      if (response.ok) break;
      if (!data.retryable || attempt === 29) throw Error(data.error || 'Server could not serve the request.');
      const after = Number(response.headers.get('Retry-After'));
      const seconds = Math.max(3, Math.min(30, Number.isFinite(after) && after > 0 ? after : 8));
      statusEl.textContent = 'Model capacity unavailable. Retrying in ' + seconds + ' seconds…';
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    }
    showChatMode(data.mode);
    messages = [...outgoing, { role: 'assistant', content: data.reply }];
    if(!assistantBubble)bubble('assistant', data.reply);
    statusEl.textContent = '';
  } catch (error) {
    userBubble?.remove();
    assistantBubble?.remove();
    if (!messages.length) document.getElementById('welcome').hidden = false;
    prompt.value = text;
    resize();
    statusEl.textContent = error.message;
  } finally {
    window.axonSecurityReset();
    busy = false;
    send.disabled = false;
    document.getElementById('new').disabled = false;
    prompt.focus();
  }
};
function showChatMode(mode) {
  document.getElementById('chat-mode').textContent =
    mode === 'assisted' ? '· pretrained assistance' :
    mode === 'learning' ? '· student still training' :
    mode === 'student' ? '· evaluated student' : '';
}
// The actual mode is shown after the chat reply, not guessed from status.
