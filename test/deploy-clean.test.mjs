import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import chat from '../api/chat.js';
import status from '../api/status.js';

function response(){return {code:200,headers:{},body:null,setHeader(k,v){this.headers[k]=v;return this},status(n){this.code=n;return this},json(x){this.body=x;return this}}}

// Vercel's published root must be stateless, contain no large pretrained weights,
// and never import the old on-device model runtime.
test('Deployment has no old pretrained weights or runtime',()=>{
 const names=[];
 function walk(folder){for(const e of fs.readdirSync(folder,{withFileTypes:true})){
   if(e.name==='node_modules'||e.name==='.git')continue;
   const p=path.join(folder,e.name);
   if(e.isDirectory())walk(p);else names.push(p);
 }}
 walk('server');walk('api');walk('public');
 assert.ok(!names.some(n=>/\.(onnx|safetensors|gguf|pt|pth)$/i.test(n)));
 assert.equal(fs.existsSync('server/model.js'),false);
 assert.equal(fs.existsSync('server/hosted.js'),false);
 const pkg=JSON.parse(fs.readFileSync('package.json','utf8'));
 assert.ok(!Object.keys(pkg.dependencies||{}).some(n=>/transformers|torch|onnx/i.test(n)));
});

test('Chat missing gateway credentials returns actionable error without pretending success',async()=>{
 const old=process.env.AI_GATEWAY_API_KEY,oldOidc=process.env.VERCEL_OIDC_TOKEN,oldV=process.env.VERCEL;
 try{
   delete process.env.AI_GATEWAY_API_KEY;delete process.env.VERCEL_OIDC_TOKEN;delete process.env.VERCEL;
   const res=response();
   await chat({method:'POST',headers:{},body:{messages:[{role:'user',content:'hello again'}]}},res);
   assert.equal(res.code,503);
   assert.equal(res.body.code,'auth_missing');
   assert.match(res.body.error,/temporarily unavailable/);
   assert.equal(res.body.reply,undefined);
 }finally{
   if(old===undefined)delete process.env.AI_GATEWAY_API_KEY;else process.env.AI_GATEWAY_API_KEY=old;
   if(oldOidc===undefined)delete process.env.VERCEL_OIDC_TOKEN;else process.env.VERCEL_OIDC_TOKEN=oldOidc;
   if(oldV===undefined)delete process.env.VERCEL;else process.env.VERCEL=oldV;
 }
});

test('Status shows hosted model, never an installed local model',async()=>{
 const res=response();await status({method:'GET',headers:{}},res);
 assert.equal(res.code,200);
 assert.equal(res.body.local_model_installed,false);
 assert.equal(typeof res.body.chat_credentials_present,'boolean');
});
