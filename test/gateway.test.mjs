import test from 'node:test';
import assert from 'node:assert/strict';
import {gatewayGenerate,gatewayToken,GatewayError,classifyGatewayError} from '../server/gateway.js';

const request={headers:{}};
test('Only server-provided AI Gateway credentials are considered',()=>{
  assert.equal(gatewayToken(request,{AI_GATEWAY_API_KEY:'test-key'}),'test-key');
  assert.equal(gatewayToken(request,{VERCEL_OIDC_TOKEN:'oidc-test'}),'oidc-test');
  assert.equal(gatewayToken(request,{}),null);
  assert.equal(gatewayToken({headers:{'x-vercel-oidc-token':'browser-supplied-token'}},{}),null);
});

test('Official AI SDK receives chat with model name and returns a reply',async()=>{
  let called=0;
  const answer=await gatewayGenerate([{role:'user',content:'hello there'}],request,{
    env:{VERCEL:'1'},
    generateTextFn:async(options)=>{
      called++;
      assert.equal(options.model,'openai/gpt-4.1-mini');
      assert.equal(options.messages.at(-1).content,'hello there');
      assert.equal(options.messages[0].role,'system');
      assert.equal(options.maxOutputTokens,450);
      return {text:'Hello!'};
    }
  });
  assert.equal(called,1);assert.equal(answer,'Hello!');
});

test('On Vercel, SDK may use automatic OIDC when no API key is present',async()=>{
  const answer=await gatewayGenerate([{role:'user',content:'hi'}],request,{
    env:{VERCEL:'1'},generateTextFn:async()=>({text:'Hi!'}),
  });
  assert.equal(answer,'Hi!');
});

test('Local execution without Vercel or a key returns configuration error',async()=>{
  await assert.rejects(()=>gatewayGenerate([{role:'user',content:'hello'}],request,{env:{}}),e=>e instanceof GatewayError&&e.code==='auth_missing');
});

test('Provider errors classify billing, key failure and request failures without exposing secrets',()=>{
  assert.equal(classifyGatewayError({statusCode:401,message:'secret'}).code,'auth_rejected');
  assert.equal(classifyGatewayError({statusCode:402}).code,'billing');
  assert.equal(classifyGatewayError({statusCode:403}).code,'access_denied');
  assert.equal(classifyGatewayError({statusCode:429}).code,'rate_limited');
  assert.equal(classifyGatewayError({statusCode:504}).code,'timeout');
  assert.doesNotMatch(classifyGatewayError({statusCode:401,message:'secret'}).message,/secret/);
});
