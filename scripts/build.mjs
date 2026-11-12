import fs from 'node:fs';
import path from 'node:path';

// Prevent accidentally publishing the old Vercel model download backend.
const obsoletePaths = ['server/model.js','vercel/model.js','api/model.py'];
const stale = obsoletePaths.filter(name => fs.existsSync(name));
if (stale.length) {
  throw new Error('Remove obsolete model-loading files before deploying: '+stale.join(', '));
}
const lock=JSON.parse(fs.readFileSync('package-lock.json','utf8'));
const deps={...lock.packages?.['']?.dependencies,...lock.packages?.['']?.optionalDependencies};
if (deps?.['@xenova/transformers'] || deps?.['@huggingface/transformers'] || deps?.['onnxruntime-node']) {
  throw new Error('The legacy local-model runtime is still in package-lock.json.');
}
fs.mkdirSync('public',{recursive:true});
fs.writeFileSync('public/config.js','window.AXON_API="";\n');
console.log('AXON AI hosted-chat deployment ready (no downloaded AI weights).');
