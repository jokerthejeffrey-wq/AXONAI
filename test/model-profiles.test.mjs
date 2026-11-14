import status from '../api/status.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import {EXPANDED_ARCHITECTURE,LEGACY_ARCHITECTURE,parameterCount,checkpointBytes} from '../server/model-profiles.js';
import {MANIFEST} from '../server/train-core.js';

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
