import test from 'node:test';
import assert from 'node:assert/strict';
import {beginGateway,gatewayDeltas} from '../server/gateway-stream.js';
import {readFileSync} from 'node:fs';

test('Cloud provider never runs when no key is configured',async()=>{
 const old=process.env.AI_GATEWAY_API_KEY;delete process.env.AI_GATEWAY_API_KEY;
 try{assert.equal(await beginGateway([{role:'user',content:'hi'}],{fetcher:()=>{throw Error('Unexpected network');}}),null);}
 finally{if(old===undefined)delete process.env.AI_GATEWAY_API_KEY;else process.env.AI_GATEWAY_API_KEY=old;}
});

test('Gateway SSE chunks stream actual decoded subword content',async()=>{
 const lines=[
   'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
   'data: {"choices":[{"delta":{"content":" there!"}}]}\n\n',
   'data: [DONE]\n\n'
 ];
 const body=new ReadableStream({
  start(c){for(const line of lines){const bytes=new TextEncoder().encode(line);c.enqueue(bytes.slice(0,7));c.enqueue(bytes.slice(7));}c.close();}
 });
 const parts=[];
 for await(const delta of gatewayDeltas({body}))parts.push(delta);
 assert.deepEqual(parts,['Hello',' there!']);
});
test('Browser worker uses a true tokenizer and weight-backed generation, not fixed greetings',()=>{
 const source=readFileSync('public/static/local-model-worker.js','utf8');
 assert.match(source,/TextStreamer/);
 assert.match(source,/generator\.tokenizer/);
 assert.match(source,/pipeline\('text-generation'/);
 assert.match(source,/onnx-community/);
 assert.doesNotMatch(source,/My student model is still training/);
 const ui=readFileSync('public/static/chat.js','utf8');
 assert.match(ui,/new Worker\(/);
 assert.doesNotMatch(ui,/Hello! I'm AXON AI/);
});
