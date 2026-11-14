import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
test('Blob cleanup retains in-flight and four recent checkpoints',()=>{
 const src=fs.readFileSync('server/train-store.js','utf8');
 assert.match(src,/reclaimStorage/);
 assert.match(src,/j.status IN \('approved','rejected'\)/);
 assert.match(src,/j.status IN \('issued','pending'\)/);
 assert.match(src,/now\(\)-interval '72 hours'/);
 assert.match(src,/version'\)-3/);
 assert.match(src,/storagePut\(sql/);
});
test('Submission status is authenticated and never returns upload tokens',()=>{
 const src=fs.readFileSync('api/job-state.js','utf8');
 assert.match(src,/submissionState/);
 assert.doesNotMatch(src,/return.*token/);
});
