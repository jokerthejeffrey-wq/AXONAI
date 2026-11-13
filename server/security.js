// Request checks and optional CAPTCHA verification for the public chat API.
// The local counters below are per Vercel instance, not distributed quotas.
const WINDOW_MS = 60_000;
const MAX_TRACKED_CLIENTS = 2500;
const requestBuckets = new Map();
const generationBuckets = new Map();

function clientId(req) {
  const headers = req.headers || {};
  // Vercel's forwarded address is used when available. This is only a
  // best-effort limiter; enforce global quotas at the Vercel Firewall edge.
  const value = headers['x-vercel-forwarded-for'] || headers['x-forwarded-for'] || headers['x-real-ip'] || req.socket?.remoteAddress || 'unknown';
  return String(Array.isArray(value) ? value[0] : value).split(',')[0].trim().slice(0, 128);
}

function checkBucket(bucketMap, id, limit, now) {
  if (bucketMap.size > MAX_TRACKED_CLIENTS) {
    for (const [key, value] of bucketMap) if (value.until <= now) bucketMap.delete(key);
    while (bucketMap.size > MAX_TRACKED_CLIENTS) bucketMap.delete(bucketMap.keys().next().value);
  }
  let bucket = bucketMap.get(id);
  if (!bucket || bucket.until <= now) {
    bucket = { count: 0, until: now + WINDOW_MS };
    bucketMap.set(id, bucket);
  }
  if (bucket.count >= limit) return Math.max(1, Math.ceil((bucket.until - now) / 1000));
  bucket.count++;
  return 0;
}

export function rateLimit(req, kind = 'requests', now = Date.now()) {
  return checkBucket(kind === 'generations' ? generationBuckets : requestBuckets,
    clientId(req), kind === 'generations' ? 8 : 120, now);
}

export function originAllowed(req) {
  const h = req.headers || {};
  const site = String(h['sec-fetch-site'] || '').toLowerCase();
  if (site && site !== 'same-origin' && site !== 'none') return false;
  if (!h.origin) return true; // HTTP clients omit Origin; Turnstile/WAF protects production.
  const host = String(h.host || '').toLowerCase();
  if (!host) return false;
  try {
    const origin = new URL(String(h.origin));
    return origin.host.toLowerCase() === host &&
      (origin.protocol === 'https:' || (origin.protocol === 'http:' && /^(localhost|127\.0\.0\.1)(:\d+)?$/i.test(host)));
  } catch { return false; }
}

export function parseChatBody(req) {
  const type = String(req.headers?.['content-type'] || '').toLowerCase();
  if (type && !/^application\/json(?:\s*;|\s*$)/.test(type)) {
    const err = new Error('Content-Type must be application/json.'); err.status = 415; throw err;
  }
  const declared = Number(req.headers?.['content-length'] || 0);
  if (declared > 131072) { const err = new Error('Request is too large.'); err.status = 413; throw err; }
  const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  if (!body || typeof body !== 'object' || Array.isArray(body) ||
      Object.keys(body).some(k => !['messages', 'turnstileToken', 'stream'].includes(k))) throw Error('Invalid request body.');
  if (Buffer.byteLength(JSON.stringify(body), 'utf8') > 131072) {
    const err = new Error('Request is too large.'); err.status = 413; throw err;
  }
  if (body.turnstileToken !== undefined &&
      (typeof body.turnstileToken !== 'string' || body.turnstileToken.length > 2048)) throw Error('Invalid verification token.');
  if(body.stream!==undefined && typeof body.stream!=='boolean')throw Error('Invalid stream option.');
  return body;
}

export function turnstileConfig() {
  const siteKey = process.env.AXON_TURNSTILE_SITE_KEY?.trim() || '';
  const secretKey = process.env.AXON_TURNSTILE_SECRET_KEY?.trim() || '';
  return { enabled: Boolean(siteKey && secretKey), misconfigured: Boolean(siteKey) !== Boolean(secretKey), siteKey, secretKey };
}

export async function verifyTurnstile(req, token, config = turnstileConfig(), fetcher = fetch) {
  if (config.misconfigured) return { ok: false, status: 503, error: 'Security verification is not configured correctly.' };
  if (!config.enabled) return { ok: true };
  if (typeof token !== 'string' || !token || token.length > 2048) {
    return { ok: false, status: 403, error: 'Complete the security verification and retry.' };
  }
  try {
    const params = new URLSearchParams({ secret: config.secretKey, response: token });
    const ip = clientId(req);
    if (ip !== 'unknown') params.set('remoteip', ip);
    const response = await fetcher('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
      method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params, signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw Error('Verification service failed');
    const result = await response.json();
    const requestHost = String(req.headers?.host || '').toLowerCase();
    const actualHost = String(result.hostname || '').toLowerCase();
    // Cloudflare test keys return a synthetic hostname; useful for local tests.
    const testKey = config.siteKey.startsWith('1x00000000000000000000');
    if (result.success === true && result.action === 'axon_chat' &&
        (testKey || (actualHost && requestHost && actualHost === requestHost))) return { ok: true };
    return { ok: false, status: 403, error: 'Security verification failed. Try again.' };
  } catch {
    return { ok: false, status: 503, error: 'Security verification is temporarily unavailable.' };
  }
}
