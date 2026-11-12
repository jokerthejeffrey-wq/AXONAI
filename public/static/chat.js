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
  let userBubble = null;
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
        body: JSON.stringify({ messages: outgoing, ...(token ? { turnstileToken: token } : {}) })
      });
      data = await response.json();
      if (response.ok) break;
      if (!data.retryable || attempt === 29) throw Error(data.error || 'Server could not serve the request.');
      const after = Number(response.headers.get('Retry-After'));
      const seconds = Math.max(3, Math.min(30, Number.isFinite(after) && after > 0 ? after : 8));
      statusEl.textContent = 'Model capacity unavailable. Retrying in ' + seconds + ' seconds…';
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    }
    showChatMode();
    messages = [...outgoing, { role: 'assistant', content: data.reply }];
    bubble('assistant', data.reply);
    statusEl.textContent = '';
  } catch (error) {
    userBubble?.remove();
    prompt.value = text;
    resize();
    statusEl.textContent = error.message + ' Your message is ready to retry.';
  } finally {
    window.axonSecurityReset();
    busy = false;
    send.disabled = false;
    document.getElementById('new').disabled = false;
    prompt.focus();
  }
};
function showChatMode() { document.getElementById('chat-mode').textContent = ''; }
axonFetch('/api/status').then(r => { if (!r.ok) throw Error(); return r.json(); })
  .then(showChatMode).catch(() => { document.getElementById('chat-mode').textContent = ''; });
