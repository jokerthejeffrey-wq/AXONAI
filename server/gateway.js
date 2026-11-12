// The chatbot is hosted through Vercel AI Gateway. No model weights are
// installed, downloaded, or persisted inside Vercel Functions.
const GATEWAY_URL = 'https://ai-gateway.vercel.sh/v1/chat/completions';
const DEFAULT_MODEL = 'openai/gpt-4.1-mini';

export class GatewayError extends Error {
  constructor(message, code='upstream', status=503, retryable=false) {
    super(message);
    this.name = 'GatewayError';
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

export function gatewayToken(req, env=process.env) {
  const apiKey = typeof env.AI_GATEWAY_API_KEY === 'string' ? env.AI_GATEWAY_API_KEY.trim() : '';
  if (apiKey) return apiKey;
  const oidc = typeof env.VERCEL_OIDC_TOKEN === 'string' ? env.VERCEL_OIDC_TOKEN.trim() : '';
  if (oidc) return oidc;
  // Vercel Functions can receive a provisioned OIDC token in this header.
  // Only accept this header when running on Vercel, not from local callers.
  if (env.VERCEL) {
    const header = req?.headers?.['x-vercel-oidc-token'];
    if (typeof header === 'string' && header.length > 20) return header.trim();
  }
  return null;
}

export function gatewayModel(env=process.env) {
  const model = (env.AXON_GATEWAY_MODEL || DEFAULT_MODEL).trim();
  if (!/^[a-z0-9-]+\/[a-zA-Z0-9._:-]+$/.test(model)) {
    throw new GatewayError('Invalid model configuration.', 'config', 503);
  }
  return model;
}

export async function gatewayGenerate(messages, req, {fetcher=fetch,env=process.env}={}) {
  const token = gatewayToken(req,env);
  if (!token) {
    throw new GatewayError('AI Gateway credentials are missing. Enable AI Gateway for this Vercel project.', 'auth_missing', 503);
  }
  const model = gatewayModel(env);
  let response;
  try {
    response = await fetcher(GATEWAY_URL, {
      method: 'POST',
      headers: {Authorization:`Bearer ${token}`, 'Content-Type':'application/json'},
      body: JSON.stringify({
        model,
        messages: [
          {role:'system', content:'You are AXON AI, a helpful community-built AI project assistant. Answer correctly, clearly, and concisely. You are powered by pretrained assistance while a separate AXON student model is trained. Never claim the student model already has your capabilities. Say when you are unsure.'},
          ...messages.slice(-11)
        ],
        max_tokens:500,
        temperature:0.3,
        stream:false,
      }),
      signal:AbortSignal.timeout(40000),
    });
  } catch (error) {
    if (error?.name === 'TimeoutError' || error?.name === 'AbortError') {
      throw new GatewayError('Chat provider timed out.', 'timeout', 504, false);
    }
    throw new GatewayError('Chat provider could not be reached.', 'network', 503, false);
  }
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new GatewayError('AI Gateway rejected authentication. Check project AI Gateway access.', 'auth_rejected', 503);
    }
    if (response.status === 402) {
      throw new GatewayError('AI Gateway credits or billing are unavailable.', 'billing', 503);
    }
    if (response.status === 429) {
      throw new GatewayError('The AI provider is busy. Please try again shortly.', 'rate_limited', 503);
    }
    if (response.status === 404 || response.status === 400) {
      throw new GatewayError('AI Gateway model settings are invalid.', 'model_unavailable', 503);
    }
    throw new GatewayError(`AI Gateway returned HTTP ${response.status}.`, 'upstream', 503);
  }
  let data;
  try { data = await response.json(); }
  catch { throw new GatewayError('AI Gateway returned an invalid response.', 'invalid_response', 503); }
  const content = data?.choices?.[0]?.message?.content;
  const answer = typeof content === 'string' ? content
    : Array.isArray(content) ? content.filter(v => v?.type==='text').map(v => v.text||'').join('\n')
    : '';
  if (!answer.trim()) throw new GatewayError('AI Gateway returned no reply.', 'empty_reply', 503);
  return answer.trim();
}
