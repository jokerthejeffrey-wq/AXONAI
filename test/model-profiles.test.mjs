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
