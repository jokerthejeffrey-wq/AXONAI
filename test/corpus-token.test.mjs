import test from 'node:test';
import assert from 'node:assert/strict';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {lesson,makeLessons,CURRICULUM_VERSION} from '../server/train-core.js';
import {sampleNextToken} from '../server/scratch-infer.js';

test('same curriculum version produces different examples for different global indices',()=>{
 assert.equal(CURRICULUM_VERSION,3);
 const seen=new Set();
 for(let i=0;i<16000;i++){
  const row=lesson(i);
  const key=row.messages[0].content+'\u0000'+row.response;
  assert.ok(!seen.has(key),'Duplicate lesson at index '+i);
  seen.add(key);
  assert.equal(row.messages[0].role,'user');
  assert.ok(row.response.length>0 && row.response.length<500);
 }
 assert.equal(makeLessons(32).length,8);
 for(const i of [1,10_000,15_000_000,22_000_000]){
  const r=lesson(i);
  assert.equal(typeof r.response,'string');
 }
 assert.throws(()=>lesson(-1),/Invalid/);
});
test('token probabilities are sampled, not always greedy argmax',()=>{
 const logits=new Float32Array(260).fill(-12);
 logits[65]=5;logits[66]=5;
 assert.equal(sampleNextToken(logits,{rng:()=>0,topK:2,topP:1,step:10}),65);
 assert.equal(sampleNextToken(logits,{rng:()=>0.99999,topK:2,topP:1,step:10}),66);
 assert.throws(()=>sampleNextToken(logits,{temperature:0}),/Invalid/);
});
test('corpus generator exports the exact lessons sent to training clients',()=>{
 const folder=mkdtempSync(join(tmpdir(),'axon-corpus-'));
 try{
  const out=join(folder,'corpus.txt');
  execFileSync(process.execPath,['scripts/generate-corpus.mjs',out,'256'],{timeout:20000});
  const lines=readFileSync(out,'utf8').trimEnd().split('\n');
  assert.equal(lines.length,256);
  for(let i=0;i<lines.length;i++){
    const row=lesson(i);
    assert.equal(lines[i],row.messages[0].content+'\t'+row.response);
  }
 }finally{rmSync(folder,{force:true,recursive:true})}
});
test('public 1 GB corpus generator is available',()=>{
 const src=readFileSync('public/downloads/AXON-Corpus-1GB.py','utf8');
 assert.match(src,/15_000_001/);
 assert.match(src,/1_073_741_824/);
 assert.match(src,/def lesson\(n\)/);
});
