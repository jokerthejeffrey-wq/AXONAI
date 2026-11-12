import { turnstileConfig } from '../server/security.js';
export default function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'GET') { res.setHeader('Allow', 'GET'); return res.status(405).json({ error: 'Use GET.' }); }
  const config = turnstileConfig();
  if (config.misconfigured) return res.status(503).json({ error: 'Security verification is not configured correctly.' });
  return res.status(200).json({ turnstileEnabled: config.enabled, siteKey: config.enabled ? config.siteKey : null });
}
