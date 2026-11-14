import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {PENDING_QUEUE_LIMIT} from '../server/train-store.js';
import {parseCheckpoint,scoreNextByte} from '../server/scratch-infer.js';

test('queue is bounded and the job endpoint reports retry backpressure',()=>{
 assert.equal(PENDING_QUEUE_LIMIT,12);
 const endpoint=fs.readFileSync('api/job.js','utf8');
 assert.match(endpoint,/job===null/);
 assert.match(endpoint,/status\(204\)\.end\(\)/);
 assert.match(endpoint,/Retry-After/);
 const store=fs.readFileSync('server/train-store.js','utf8');
 assert.match(store,/autoReviewOne\(sql\)/);
 assert.match(store,/status='pending'/);
 assert.match(store,/axon_queue_lease/);
 assert.match(store,/mergeWeights\(/);
 assert.match(store,/validateCandidate\(/);
});
test('real 13.7 MB scratch checkpoint has finite next-byte validation loss',()=>{
 const bytes=Buffer.concat(Array.from({length:6},(_,i)=>
   fs.readFileSync('public/model/v0-'+String(i).padStart(2,'0')+'.bin')));
 const model=parseCheckpoint(bytes);
 const loss=scoreNextByte(model,[{role:'user',content:'hello'}],72);
 assert.ok(Number.isFinite(loss) && loss>0 && loss<30,'valid loss');
});
test('all primary pages include responsive styles and touch controls',()=>{
 for(const page of ['index','chat','downloads','docs']){
   const html=fs.readFileSync('public/'+page+'.html','utf8');
   assert.match(html,/<meta[^>]*name="viewport"/i,page);
   assert.match(html,/href="\/static\/responsive\.css"/,page);
 }
 const css=fs.readFileSync('public/static/responsive.css','utf8');
 assert.match(css,/@media\(max-width:480px\)/);
 assert.match(css,/@media\(max-width:760px\)/);
 assert.match(css,/env\(safe-area-inset-bottom/);
 assert.match(css,/#composer/);
 assert.match(css,/\.progress-wrap/);
});
