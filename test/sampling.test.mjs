import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {sampleToken} from '../server/sampling.js';
import {scratchStream,parseCheckpoint} from '../server/scratch-infer.js';
test('Sampling can choose different tokens; special roles are suppressed',()=>{
 const logits=new Float32Array(260).fill(-100);logits[65]=1;logits[66]=1;logits[257]=100;
 assert.equal(sampleToken(logits,[],{random:()=>0}),65);
 assert.equal(sampleToken(logits,[],{random:()=>.99}),66);
 assert.equal(sampleToken(logits,[65],{random:()=>0}),66);
});
test('Actual checkpoint streams text chunks without waiting for entire reply',async()=>{
 const bytes=Buffer.concat(Array.from({length:6},(_,i)=>fs.readFileSync(`public/model/v0-${String(i).padStart(2,'0')}.bin`)));
 let count=0;for await(const delta of scratchStream(parseCheckpoint(bytes),[{role:'user',content:'Hi'}],{maxNewTokens:6,random:()=>0})){assert.equal(typeof delta,'string');count++;}
 assert.ok(count>0);
});

import {parseChatBody} from '../server/security.js';
test('Streaming flag passes security validation only as boolean',()=>{
 assert.equal(parseChatBody({headers:{},body:{messages:[],stream:true}}).stream,true);
 assert.throws(()=>parseChatBody({headers:{},body:{messages:[],stream:'true'}}));
});
