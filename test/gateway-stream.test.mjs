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
test('Public browser uses only AXON scratch checkpoints',()=>{
 const ui=readFileSync('public/static/chat.js','utf8');
 const server=readFileSync('api/chat.js','utf8');
 assert.match(ui,/experimental-preview/);
 assert.match(server,/scratchStream/);
 assert.match(server,/loadStudent/);
 assert.equal(ui.includes('new Worker('),false);
 assert.doesNotMatch(ui,/generateInBrowser|generateFromCloud/);
 assert.doesNotMatch(server,/beginGateway|browser_model_required|Qwen/);
});
