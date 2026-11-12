// Optional challenge configuration is served by AXONAI's same-origin API.
let axonTurnstileWidget = null;
let axonTurnstileToken = '';
let axonSecurityEnabled = false;

function axonLoadTurnstile() {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(Error('Security verification could not load. Check your connection.'));
    document.head.append(script);
  });
}

window.axonSecurityReady = (async () => {
  const response = await axonFetch('/api/security');
  if (!response.ok) throw Error('Security verification is unavailable. Please try again later.');
  const cfg = await response.json();
  if (!cfg.turnstileEnabled) return;
  axonSecurityEnabled = true;
  await axonLoadTurnstile();
  if (!window.turnstile) throw Error('Security verification did not initialize.');
  axonTurnstileWidget = window.turnstile.render('#turnstile-container', {
    sitekey: cfg.siteKey,
    action: 'axon_chat',
    theme: 'auto',
    callback: token => { axonTurnstileToken = token; },
    'expired-callback': () => { axonTurnstileToken = ''; },
    'error-callback': () => { axonTurnstileToken = ''; },
  });
})();

window.axonSecurityToken = () => {
  if (!axonSecurityEnabled) return null;
  if (!axonTurnstileToken) throw Error('Complete security verification before sending.');
  return axonTurnstileToken;
};

window.axonSecurityReset = () => {
  if (axonSecurityEnabled && axonTurnstileWidget !== null && window.turnstile) {
    axonTurnstileToken = '';
    window.turnstile.reset(axonTurnstileWidget);
  }
};
