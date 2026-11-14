import test from 'node:test';
import assert from 'node:assert/strict';
import {pacedWords,countWords,draftScore,chooseDraft} from '../server/deliberate-chat.js';

test('A response preserves formatting and waits at least 500ms before each word',async()=>{
  const waits=[],pieces=[];
  for await (const piece of pacedWords('Hello,  world!\nAgain.',{wait:async ms=>waits.push(ms)}))
    pieces.push(piece);
  assert.equal(pieces.join(''),'Hello,  world!\nAgain.');
  assert.deepEqual(waits,[500,500,500]);
  assert.equal(countWords(pieces.join('')),3);
});

test('A longer configured interval is respected, but a shorter one is rejected',async()=>{
  const waits=[];
  for await (const _ of pacedWords('Two words',{minimumMs:100,wait:async ms=>waits.push(ms)})) {}
  assert.deepEqual(waits,[500,500]);
  waits.length=0;
  for await (const _ of pacedWords('Two words',{minimumMs:750,wait:async ms=>waits.push(ms)})) {}
  assert.deepEqual(waits,[750,750]);
});

test('Candidate review favors coherence and topic while never inventing answers',()=>{
  const messages=[{role:'user',content:'How can I build a robot with distance sensors?'}];
  const good='A robot can use distance sensors to notice obstacles and avoid collisions.';
  const bad='blah blah blah blah blah blah blah blah blah blah blah';
  assert.ok(draftScore(good,messages)>draftScore(bad,messages));
  assert.equal(chooseDraft([bad,good],messages),good);
  assert.equal(chooseDraft(['',bad],messages,{accept: s=>s.includes('sensor')}),'');
});
