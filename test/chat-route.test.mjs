import test from 'node:test';
import assert from 'node:assert/strict';
import chat from '../api/chat.js';
function response(){return {code:200,headers:{},body:null,setHeader(k,v){this.headers[k]=v;return this},status(n){this.code=n;return this},json(x){this.body=x;return this}}}
test('Real API route returns hosted assistant reply with mocked gateway',async()=>{
  const prior=process.env.AI_GATEWAY_API_KEY;
  const priorFetch=globalThis.fetch;
  const priorHosted=process.env.AXON_OPENROUTER_KEY;
  try {
    process.env.AI_GATEWAY_API_KEY='test-private-key';
    delete process.env.AXON_OPENROUTER_KEY;
    globalThis.fetch=async (url,opts)=>{
      assert.equal(url,'https://ai-gateway.vercel.sh/v1/chat/completions');
      assert.equal(opts.headers.Authorization,'Bearer test-private-key');
      return {ok:true,json:async()=>({choices:[{message:{content:'Hi from AXON AI'}}]})};
    };
    const res=response();
    await chat({method:'POST',headers:{},body:{messages:[{role:'user',content:'hello there'}]}},res);
    assert.equal(res.code,200);
    assert.equal(res.body.reply,'Hi from AXON AI');
    assert.equal(res.body.mode,'pretrained');
  }finally{
    globalThis.fetch=priorFetch;
    if(prior===undefined)delete process.env.AI_GATEWAY_API_KEY;else process.env.AI_GATEWAY_API_KEY=prior;
    if(priorHosted===undefined)delete process.env.AXON_OPENROUTER_KEY;else process.env.AXON_OPENROUTER_KEY=priorHosted;
  }
});
