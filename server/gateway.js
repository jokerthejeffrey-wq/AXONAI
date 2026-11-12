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

export function gatewayToken(req, env = process.env) {
  const key = typeof env.AI_GATEWAY_API_KEY === 'string' ? env.AI_GATEWAY_API_KEY.trim() : '';
  if (key) return key;
  const oidc = typeof env.VERCEL_OIDC_TOKEN === 'string' ? env.VERCEL_OIDC_TOKEN.trim() : '';
  if (oidc) return oidc;
  if (env.VERCEL) {
    // Vercel provides its fresh OIDC token through this header to functions.
    const header = req?.headers?.['x-vercel-oidc-token'];
    if (typeof header === 'string' && header.length > 20) return header.trim();
  }
  return null;
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
  if (code.includes('api key') || code.includes('no credentials')) return new GatewayError('AI Gateway authentication is missing.','auth_missing');
  return new GatewayError('AI Gateway is temporarily unavailable.','upstream');
}

export async function gatewayGenerate(messages, req, {env = process.env, generateTextFn} = {}) {
  const model = gatewayModel(env);
  // Local tests should not access a paid provider. On Vercel the AI SDK
  // can use its automatic OIDC mechanism even when there is no API key.
  if (!generateTextFn && !env.VERCEL && !gatewayToken(req, env)) {
    throw new GatewayError('Enable Vercel AI Gateway or configure a server-side API key.','auth_missing');
  }
  const chat = [
    {role:'system',content:SYSTEM},
    ...messages.slice(-11).map(m=>({role:m.role,content:m.content}))
  ];
  try {
    const sdk = generateTextFn ? null : await import('ai');
    const generateText = generateTextFn || sdk.generateText;
    // Vercel injects a short-lived project OIDC token into each function
    // request. Prefer it when available, even if the owner accidentally set
    // an invalid manual API key. This keeps credentials out of the ZIP.
    const oidcHeader = env.VERCEL && typeof req?.headers?.['x-vercel-oidc-token'] === 'string'
      ? req.headers['x-vercel-oidc-token'].trim() : '';
    const projectOidc = env.VERCEL ? (oidcHeader || env.VERCEL_OIDC_TOKEN || '') : '';
    const modelProvider = !generateTextFn && projectOidc && sdk.createGateway
      ? sdk.createGateway({apiKey:projectOidc})(model) : model;
    const options = {
      model:modelProvider,
      messages:chat,
      temperature:0.3,
      maxOutputTokens:450,
      abortSignal:AbortSignal.timeout(45000),
    };
    let result;
    try { result = await generateText(options); }
    catch (error) {
      // A valid manually configured key is the backup if project OIDC is
      // unavailable. Do not retry for billing or transient provider errors.
      const failure=classifyGatewayError(error);
      if (modelProvider === model || !env.AI_GATEWAY_API_KEY || !['auth_rejected','access_denied'].includes(failure.code)) throw error;
      result = await generateText({...options,model});
    }
    const answer = typeof result?.text === 'string' ? result.text.trim() : '';
    if (!answer) throw new GatewayError('The model returned an empty response.','empty_reply');
    return answer;
  } catch (error) {
    throw classifyGatewayError(error);
  }
}
