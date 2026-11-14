import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
test('Vercel build configuration is valid JSON and permits model downloads',()=>{
  const config=JSON.parse(readFileSync('vercel.json','utf8'));
  const policy=config.headers[0].headers.find(x=>x.key==='Content-Security-Policy').value;
  assert.match(policy,/https:\/\/huggingface.co/);
  assert.match(policy,/https:\/\/cdn.jsdelivr.net/);
  assert.match(policy,/worker-src 'self'/);
  assert.match(policy,/wasm-unsafe-eval/);
});
