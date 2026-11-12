import test from 'node:test';
import assert from 'node:assert/strict';
import chat from '../api/chat.js';
function response(){return {code:200,headers:{},body:null,setHeader(k,v){this.headers[k]=v;return this},status(n){this.code=n;return this},json(x){this.body=x;return this}}}
test('Real API chat route answers simple calculations without provider access',async()=>{
  const res=response();
  await chat({method:'POST',headers:{},body:{messages:[{role:'user',content:'100 - 100'}]}},res);
  assert.equal(res.code,200);
  assert.match(res.body.reply,/^0$/);
});
