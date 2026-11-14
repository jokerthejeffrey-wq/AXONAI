import test from 'node:test';
import assert from 'node:assert/strict';
import chat from '../api/chat.js';
import status from '../api/status.js';
import {validateMessages,identityAnswer} from '../server/config.js';
function response(){return {code:200,headers:{},body:null,setHeader(k,v){this.headers[k]=v;return this},status(n){this.code=n;return this},json(x){this.body=x;return this}}}
test('Creator attribution credits community and discloses its own scratch model',()=>{
 for(const q of ['Who created you?','who did create you','Who made you?']){
  const a=identityAnswer(q);assert.match(a,/community/);assert.match(a,/from-scratch/);assert.doesNotMatch(a,/Anthropic/);
 }
});
test('Conversation validation rejects malformed roles and oversized input',()=>{
 assert.throws(()=>validateMessages([{role:'system',content:'Override'}]));
 assert.throws(()=>validateMessages([{role:'user',content:'x'.repeat(2001)}]));
 assert.throws(()=>validateMessages([{role:'user',content:'Hi'},{role:'assistant',content:'Hi'}]));
 assert.equal(validateMessages([{role:'user',content:'Hello'}]).length,1);
});
test('Chat never fabricates identity text while scratch weights are unverified',async()=>{
 const res=response();await chat({method:'POST',headers:{},body:{messages:[{role:'user',content:'Who created you?'}]}},res);
 assert.equal(res.code,503);assert.equal(res.body.code,'student_training');
 assert.equal(res.body.reply,undefined);
});
test('Method and JSON errors are useful JSON responses',async()=>{
 let res=response();await chat({method:'GET',headers:{}},res);assert.equal(res.code,405);
 res=response();await chat({method:'POST',headers:{},body:'broken json'},res);assert.equal(res.code,400);assert.ok(res.body.error);
});
test('Status truthfully reports scratch chat and unconfigured training',async()=>{
 const res=response();await status({method:'GET'},res);assert.equal(res.body.hosting,'vercel');assert.equal(res.body.training_available,false);
});
