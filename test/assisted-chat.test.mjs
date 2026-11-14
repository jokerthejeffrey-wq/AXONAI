import test from 'node:test';
import assert from 'node:assert/strict';
import {assistedAnswer} from '../server/assisted-chat.js';

test('Pretrained assistance never runs without a configured key',async()=>{
  const old=process.env.AI_GATEWAY_API_KEY;
  delete process.env.AI_GATEWAY_API_KEY;
  try {
    const result=await assistedAnswer([{role:'user',content:'hi'}],{
      fetcher:()=>{throw Error('network called without authentication');},now:()=>0
    });
    assert.equal(result,null);
  } finally {
    if(old===undefined)delete process.env.AI_GATEWAY_API_KEY;
    else process.env.AI_GATEWAY_API_KEY=old;
  }
});

test('A verification-blocked gateway falls back without returning junk or retrying immediately',async()=>{
  const old=process.env.AI_GATEWAY_API_KEY;
  process.env.AI_GATEWAY_API_KEY='test-in-memory';
  let calls=0;
  const fetcher=async()=>{calls++;return {ok:false,status:403}};
  try {
    assert.equal(await assistedAnswer([{role:'user',content:'hi'}],{fetcher,now:()=>1000}),null);
    assert.equal(await assistedAnswer([{role:'user',content:'hi'}],{fetcher,now:()=>2000}),null);
    assert.equal(calls,1);
  } finally {
    if(old===undefined)delete process.env.AI_GATEWAY_API_KEY;
    else process.env.AI_GATEWAY_API_KEY=old;
  }
});

test('A successful pretrained response is plain text without leaking credentials',async()=>{
  const old=process.env.AI_GATEWAY_API_KEY;
  process.env.AI_GATEWAY_API_KEY='test-in-memory';
  try {
    const result=await assistedAnswer([{role:'user',content:'hi'}],{
      now:()=>1_000_000,
      fetcher:async(_url,options)=>{
        assert.equal(options.headers.Authorization,'Bearer test-in-memory');
        const body=JSON.parse(options.body);
        assert.equal(body.messages.at(-1).content,'hi');
        assert.equal(body.stream,false);
        return {ok:true,json:async()=>({choices:[{message:{content:'Hello! How can I help?'}}]})};
      }
    });
    assert.equal(result,'Hello! How can I help?');
  } finally {
    if(old===undefined)delete process.env.AI_GATEWAY_API_KEY;
    else process.env.AI_GATEWAY_API_KEY=old;
  }
});
