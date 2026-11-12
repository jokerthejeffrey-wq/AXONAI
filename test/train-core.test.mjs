import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {BATCH,MANIFEST,makeLessons,sha256,packChunks,mergeWeights} from '../server/train-core.js';
import {initialCheckpoint} from '../server/base-checkpoint.js';

test('lesson reservation sequence has no repeated prompt-answer pairs in first 4000 items',()=>{
 const keys=new Set();for(let i=0;i<500;i++)for(const row of makeLessons(i*BATCH)){
  const key=row.messages[0].content+'\0'+row.response;
  assert.ok(!keys.has(key));keys.add(key);
 }assert.equal(keys.size,4000);
});
test('manifest matches worker protocol',()=>{
 assert.equal(MANIFEST.protocol,2);
 assert.deepEqual(MANIFEST.architecture,{vocab_size:260,width:256,layers:4,heads:8,context:512});
});
test('initial model chunks reconstruct original published checksum',()=>{
 const bufs=initialCheckpoint.parts.map(p=>{
  const b=fs.readFileSync('public'+p.url);assert.equal(b.length,p.size);return b;
 });
 const whole=Buffer.concat(bufs);
 assert.equal(whole.length,initialCheckpoint.size);
 assert.equal(sha256(whole),initialCheckpoint.sha256);
 assert.equal(packChunks(whole).length,bufs.length);
});
test('unsafe checkpoint shapes and invalid updates are rejected',()=>{
 const toy=(f)=>{const header=Buffer.from(JSON.stringify({A:{dtype:'F32',shape:[2],data_offsets:[0,8]}}));const out=Buffer.alloc(8+header.length+8);out.writeBigUInt64LE(BigInt(header.length));header.copy(out,8);out.writeFloatLE(f,8+header.length);out.writeFloatLE(-f,12+header.length);return out};
 const merged=mergeWeights(toy(1),toy(1),toy(1.02));
 assert.ok(Math.abs(merged.readFloatLE(merged.length-8)-1.01)<0.00001);
 assert.throws(()=>mergeWeights(toy(1),toy(1),toy(1.5)),/excessive/);
 assert.throws(()=>mergeWeights(toy(1),toy(1),Buffer.from('bad')),/Invalid/);
});
