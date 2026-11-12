import { memoryAdmit, memoryStats } from '../server/memory.js';
import { identityAnswer, validateMessages, MODEL_LABEL } from '../server/config.js';
import { originAllowed, parseChatBody, rateLimit, turnstileConfig, verifyTurnstile } from '../server/security.js';
let busy = false;

function rejectRate(res, seconds, message) {
  res.setHeader('Retry-After', String(seconds));
  return res.status(429).json({ error: message });
}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Use POST.' });
  }
  if (!originAllowed(req)) return res.status(403).json({ error: 'This request must come from the AXONAI website.' });
  const requestWait = rateLimit(req, 'requests');
  if (requestWait) return rejectRate(res, requestWait, 'Too many requests. Please try again later.');

  let body, messages;
  try {
    body = parseChatBody(req);
    messages = validateMessages(body.messages);
  } catch (error) {
    return res.status(error.status || 400).json({ error: error.status ? error.message : 'Invalid chat request.' });
  }
  const config = turnstileConfig();
  if (config.misconfigured) return res.status(503).json({ error: 'Security verification is not configured correctly.' });
  // Check availability before consuming a single-use verification token.
  if (busy) {
    res.setHeader('Retry-After', '5');
    return res.status(429).json({ error: 'The model is busy. Waiting for capacity.', retryable: true });
  }
  if (!memoryAdmit(memoryStats(), .9, process.memoryUsage().rss < 400 * 1024 * 1024 ? 700 * 1024 * 1024 : 160 * 1024 * 1024)) {
    res.setHeader('Retry-After', '10');
    return res.status(503).json({ error: 'Server RAM is above the 90% admission threshold.', retryable: true });
  }
  const verification = await verifyTurnstile(req, body.turnstileToken, config);
  if (!verification.ok) return res.status(verification.status).json({ error: verification.error });
  const generationWait = rateLimit(req, 'generations');
  if (generationWait) return rejectRate(res, generationWait, 'Message limit reached. Please try again shortly.');
  const identity = identityAnswer(messages.at(-1).content);
  if (identity) return res.status(200).json({ reply: identity, mode: 'vercel-local', model: MODEL_LABEL });

  if (busy) {
    res.setHeader('Retry-After', '5');
    return res.status(429).json({ error: 'The model is busy. Please retry with fresh verification.' });
  }
  busy = true;
  try {
    const { generate } = await import('../server/model.js');
    const reply = await generate(messages);
    return res.status(200).json({ reply, mode: 'vercel-local', model: MODEL_LABEL });
  } catch (error) {
    console.error('AXONAI model error:', error);
    const tooLong = error.message?.startsWith('Your message is too long');
    return res.status(tooLong ? 400 : 503).json({
      error: tooLong ? error.message : 'The model could not generate a reply. Please try again later.'
    });
  } finally { busy = false; }
}
