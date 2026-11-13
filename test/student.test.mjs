import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {parseCheckpoint,scratchGenerate,encodeMessages} from '../server/scratch-infer.js';
import {scratchAnswer,localBasics,STILL_TRAINING,FALLBACK_REPLIES,readableReply} from '../server/scratch-chat.js';
import chat from '../api/chat.js';
import status from '../api/status.js';
function response(){return {code:200,headers:{},body:null,setHeader(k,v){this.headers[k]=v;return this},status(n){this.code=n;return this},json(x){this.body=x;return this}}}
function req(text){return {method:'POST',headers:{},body:{messages:[{role:'user',content:text}]}}}
test('No provider packages or Gateway authentication required',()=>{
 const pkg=JSON.parse(readFileSync('package.json','utf8'));
 assert.equal(Boolean(pkg.dependencies.ai),false);
 assert.equal(existsSync('server/gateway.js'),false);
 assert.equal(existsSync('server/hosted.js'),false);
});
test('AXON AI has local basic replies without remote calls',async()=>{
 assert.match(localBasics('hello'),/^Hello/);
 assert.match(localBasics('Explain how AI learns'),/weights/);
 const res=response();await chat(req('hello'),res);
 assert.equal(res.code,200);
 assert.equal(res.body.mode,'local-basic');
 assert.match(res.body.reply,/Hello/);
});
test('Actual scratch checkpoint parses and generates entirely in-process',()=>{
 const checkpoint=Buffer.concat(Array.from({length:6},(_,i)=>readFileSync('public/model/v0-'+String(i).padStart(2,'0')+'.bin')));
 const model=parseCheckpoint(checkpoint);
 const reply=scratchGenerate(model,[{role:'user',content:'hello'}],{maxNewTokens:6});
 assert.equal(typeof reply,'string');
 assert.deepEqual(encodeMessages([{role:'user',content:'hello'}]),[257,104,101,108,108,111,259,258]);
});
test('Untrained checkpoint never pretends to have learned answers',async()=>{
 const a=await scratchAnswer([{role:'user',content:'what is the temperature on Mars?'}],{version:0},
  {load:async()=>({}),generate:()=>'This could be a confident hallucination.'});
 assert.equal(a.mode,'learning');assert.ok(FALLBACK_REPLIES.includes(a.reply));
 assert.equal(readableReply('ñøñ%%%'),false);
});
test('Status shows no Gateway dependency',async()=>{
 const res=response();await status({method:'GET',headers:{}},res);
 assert.equal(res.code,200);
 assert.equal(res.body.chat_mode,'local-scratch');
 assert.equal(res.body.chat_requires_gateway,false);
});
