import test from 'node:test';
import assert from 'node:assert/strict';
import {memoryAdmit,memoryStats} from '../server/memory.js';

test('admit only while predicted memory remains at or below 90%',()=>{
 const total=1000;
 assert.equal(memoryAdmit({used:800,total},.9,100),true);
 assert.equal(memoryAdmit({used:801,total},.9,100),false);
 assert.equal(memoryAdmit({used:899,total},.9,2),false);
});
test('prefer Vercel cgroup memory accounting',()=>{
 const fake=(filename)=>{
  const map={'/sys/fs/cgroup/memory.current':'700','/sys/fs/cgroup/memory.max':'1000'};
  if(!(filename in map))throw Error('unavailable');
  return map[filename];
 };
 assert.deepEqual(memoryStats(fake),{used:700,total:1000,source:'cgroup'});
});
test('reject unlimited cgroup and fall back to OS stats',()=>{
 const fake=(filename)=>filename.endsWith('memory.current')?'1000':'max';
 assert.equal(memoryStats(fake).source,'system');
});
