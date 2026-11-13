// AXON AI chat uses Vercel AI Gateway through the official AI SDK.
// On Vercel, the SDK obtains project OIDC credentials automatically.  A
// server-side AI_GATEWAY_API_KEY is supported as an alternative.  No secret is
// accepted from a browser request or bundled into public assets.
const DEFAULT_MODEL = 'openai/gpt-4.1-mini';
const SYSTEM = 'You are AXON AI, a clear and accurate community-built AI assistant. Answer the user directly and correctly. The current chat uses a pretrained assistant while a separate community-trained student model is evaluated. Never claim the student already has your abilities. Admit uncertainty instead of guessing.';

export class GatewayError extends Error {
  constructor(message, code = 'upstream', status = 503, retryable = false) {
    super(message);
    this.name = 'GatewayError';
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

export function gatewayToken(_req, env = process.env) {
  // Informational only: the runtime may supply a fresh OIDC token through
  // request context. Never treat arbitrary client headers as credentials.
  const key = typeof env.AI_GATEWAY_API_KEY === 'string' ? env.AI_GATEWAY_API_KEY.trim() : '';
  if (key) return key;
  const oidc = typeof env.VERCEL_OIDC_TOKEN === 'string' ? env.VERCEL_OIDC_TOKEN.trim() : '';
  return oidc || null;
}

// Token retrieval is done inside the request (not at module load). The helper
// resolves the trusted Vercel request context and refreshes tokens when needed.
async function trustedVercelOidcToken(env, getOidcTokenFn) {
  if (!env.VERCEL && !env.VERCEL_OIDC_TOKEN) return '';
  if (getOidcTokenFn) return String((await getOidcTokenFn()) || '').trim();
  // Vercel injects a short-lived OIDC token into the function environment.
  // Use it directly before attempting an optional package import, so a
  // deployment lacking @vercel/oidc does not silently fall back to an invalid
  // manual AI_GATEWAY_API_KEY.
  const token = typeof env.VERCEL_OIDC_TOKEN === 'string' ? env.VERCEL_OIDC_TOKEN.trim() : '';
  if (token) return token;
  try {
    const {getVercelOidcToken} = await import('@vercel/oidc');
    return String((await getVercelOidcToken()) || '').trim();
  } catch {
    // The official AI SDK can also obtain OIDC automatically on Vercel.
    return '';
  }
}

export function gatewayModel(env = process.env) {
  const model = (env.AXON_GATEWAY_MODEL || DEFAULT_MODEL).trim();
  if (!/^[a-z0-9-]+\/[a-zA-Z0-9._:-]+$/.test(model)) {
    throw new GatewayError('Invalid model configuration.', 'config');
  }
  return model;
}

export function classifyGatewayError(error) {
  if (error instanceof GatewayError) return error;
  const status = Number(error?.statusCode || error?.status || error?.response?.status || 0);
  const code = String(error?.code || error?.name || '').toLowerCase();
  // Never log or expose the original upstream message; it may include data
  // from the request or service credentials.
  if (status === 401 || code.includes('authentication')) return new GatewayError('AI Gateway authentication failed.','auth_rejected');
  if (status === 402 || code.includes('payment')) return new GatewayError('AI Gateway requires credits or billing.','billing');
  if (status === 403 || code.includes('forbidden')) return new GatewayError('AI Gateway access or credits are restricted.','access_denied');
  if (status === 404 || status === 400) return new GatewayError('The selected model is not available.','model_unavailable');
  if (status === 429 || code.includes('ratelimit')) return new GatewayError('AI Gateway is busy.','rate_limited',503,true);
  if (status === 408 || status === 504 || code.includes('timeout') || code.includes('abort')) return new GatewayError('The model request timed out.','timeout',504,true);
  if (code.includes('api key') || code.includes('no credentials') || code.includes('missing_credential') || code.includes('no api key') || code.includes('missing token')) return new GatewayError('AI Gateway authentication is missing.','auth_missing');
  return new GatewayError('AI Gateway is temporarily unavailable.','upstream');
}

export async function gatewayGenerate(messages, _req, {
  env = process.env, generateTextFn, createGatewayFn, getOidcTokenFn
} = {}) {
  const model = gatewayModel(env);
  if (!generateTextFn && !env.VERCEL && !gatewayToken(null, env)) {
    throw new GatewayError('AI Gateway credentials are unavailable.','auth_missing');
  }
  const chat = [
    {role:'system', content:SYSTEM},
    ...messages.slice(-11).map(m=>({role:m.role,content:m.content}))
  ];
  try {
    const sdk = generateTextFn ? null : await import('ai');
    const generateText = generateTextFn || sdk.generateText;
    const createGateway = createGatewayFn || sdk?.createGateway;
    const oidc = await trustedVercelOidcToken(env, getOidcTokenFn);
    // An accidentally configured invalid AI_GATEWAY_API_KEY must not override
    // the trusted short-lived Vercel OIDC credential when it is available.
    const useOidc = Boolean(oidc && createGateway);
    const preferredModel = useOidc ? createGateway({apiKey:oidc})(model) : model;
    const options = {
      model:preferredModel,
      messages:chat,
      temperature:0.3,
      maxOutputTokens:450,
      abortSignal:AbortSignal.timeout(45000)
    };
    let result;
    try {
      result = await generateText(options);
    } catch (error) {
      const failure=classifyGatewayError(error);
      // A real server-side API key can be tried if the Vercel OIDC credential
      // was rejected. Other errors should not generate duplicate model calls.
      if (!useOidc || !env.AI_GATEWAY_API_KEY ||
        !['auth_rejected','access_denied'].includes(failure.code)) throw error;
      result = await generateText({...options, model});
    }
    const answer = typeof result?.text === 'string' ? result.text.trim() : '';
    if (!answer) throw new GatewayError('Empty model response.','empty_reply');
    return answer;
  } catch (error) {
    throw classifyGatewayError(error);
  }
}
