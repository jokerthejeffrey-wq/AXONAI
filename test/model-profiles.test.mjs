import {PENDING_QUEUE_LIMIT} from '../server/train-store.js';
import fs from 'node:fs';
import status from '../api/status.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import {COMPACT_ARCHITECTURE,EXPANDED_ARCHITECTURE,LEGACY_ARCHITECTURE,parameterCount,checkpointBytes} from '../server/model-profiles.js';
import {MANIFEST} from '../server/train-core.js';

test('compact research model is bounded and remains separate',()=>{
 assert.equal(parameterCount(COMPACT_ARCHITECTURE),11_240_448);
 assert.equal(checkpointBytes(COMPACT_ARCHITECTURE),44_961_792);
 assert.notDeepEqual(COMPACT_ARCHITECTURE,MANIFEST.architecture);
});

test('50M research model has verified dimensions without changing deployed manifest',()=>{
 assert.equal(parameterCount(EXPANDED_ARCHITECTURE),50_224_640);
 assert.equal(checkpointBytes(EXPANDED_ARCHITECTURE),200_898_560);
 assert.deepEqual(LEGACY_ARCHITECTURE,MANIFEST.architecture);
 assert.notDeepEqual(EXPANDED_ARCHITECTURE,MANIFEST.architecture);
 assert.throws(()=>parameterCount({...EXPANDED_ARCHITECTURE,heads:7}),/mismatch/);
});

test('Production remains on the old model while 50M research track is unactivated',async()=>{
 const res={code:200,body:null,setHeader(){return this},status(n){this.code=n;return this},json(x){this.body=x;return this}};
 await status({method:'GET'},res);
 assert.equal(res.code,200);
 assert.equal(res.body.active_model_parameters,parameterCount(LEGACY_ARCHITECTURE));
 assert.equal(res.body.experimental_model_parameters,50_224_640);
 assert.equal(res.body.experimental_model_deployed,false);
 assert.equal(res.body.five_contributors_active,false);
});

test('No daily job caps; checkpoint review queue remains bounded',()=>{
 const source=fs.readFileSync('server/train-store.js','utf8');
 assert.doesNotMatch(source,/FAIR_WORKER_JOBS_PER_DAY|DAILY_GLOBAL_JOBS|worker-daily-limit|daily-capacity-reached/);
 assert.equal(PENDING_QUEUE_LIMIT,12);
 assert.match(source,/corpus_sequence/);
});
