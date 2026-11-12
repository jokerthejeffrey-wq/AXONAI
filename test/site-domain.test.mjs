import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
test('Worker config points to AXON AI production site and no secret is bundled',()=>{
 const config=JSON.parse(fs.readFileSync('public/worker-config.json','utf8'));
 assert.equal(config.website_url,'https://axonai-theta.vercel.app');
 assert.equal(Object.keys(config).join(','),'website_url');
});
test('AI SDK authentication dependency is installed by the deployment build',()=>{
 const pkg=JSON.parse(fs.readFileSync('package.json','utf8'));
 assert.ok(pkg.dependencies.ai);
 assert.ok(!pkg.dependencies['@huggingface/transformers']);
});
