import test from 'node:test';
import assert from 'node:assert/strict';
import {gatewayGenerate, gatewayToken, GatewayError} from '../server/gateway.js';

const request={headers:{}};

test('Vercel OIDC key or private API key authenticate, never browser input',()=>{
  assert.equal(gatewayToken(request,{AI_GATEWAY_API_KEY:'k-test'}),'k-test');
  assert.equal(gatewayToken(request,{VERCEL_OIDC_TOKEN:'oidc-test'}),'oidc-test');
  assert.equal(gatewayToken(request,{}),null);
  assert.equal(gatewayToken({headers:{'x-vercel-oidc-token':'fake-long-token-not-used-on-localhost'}},{}),null);
});

test('Chat uses external AI Gateway rather than loading a model in Vercel',async()=>{
 let called=0;
 const answer=await gatewayGenerate([{role:'user',content:'hello'}],request,{
   env:{AI_GATEWAY_API_KEY:'test-server-only-token'},
   fetcher:async(url,options)=>{
     called++;
     assert.equal(url,'https://ai-gateway.vercel.sh/v1/chat/completions');
     assert.equal(options.headers.Authorization,'Bearer test-server-only-token');
     const payload=JSON.parse(options.body);
     assert.equal(payload.model,'openai/gpt-4.1-mini');
     assert.equal(payload.messages.at(-1).content,'hello');
     assert.ok(!payload.messages.some(m=>m.content.includes('test-server-only-token')));
     return {ok:true,json:async()=>({choices:[{message:{content:'Hello!'}}]})};
   }
 });
 assert.equal(called,1);assert.equal(answer,'Hello!');
});

test('Missing Vercel auth returns a configuration error instead of an endless load',async()=>{
 await assert.rejects(()=>gatewayGenerate([{role:'user',content:'hello'}],request,{env:{}}),e=>e instanceof GatewayError&&e.status===503);
});

test('Provider failures reject clearly without echoing private response',async()=>{
 await assert.rejects(()=>gatewayGenerate([{role:'user',content:'hello'}],request,{
   env:{AI_GATEWAY_API_KEY:'private'},fetcher:async()=>({ok:false,status:401,json:async()=>({secret:'do-not-expose'})})
 }),e=>e.code==='auth_rejected'&&!e.message.includes('do-not-expose'));
});
