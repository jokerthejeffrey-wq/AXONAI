import fs from 'node:fs';
import path from 'node:path';

// Only the new API endpoints import server/gateway.js. Legacy files outside this isolated
// project folder are not part of the deployed function and require no model migration.
const pkg=JSON.parse(fs.readFileSync('package.json','utf8'));
const deps={...pkg.dependencies,...pkg.optionalDependencies};
if (deps?.['@xenova/transformers'] || deps?.['@huggingface/transformers'] || deps?.['onnxruntime-node']) {
  throw new Error('The legacy local-model runtime is still in package-lock.json.');
}
fs.mkdirSync('public',{recursive:true});
fs.writeFileSync('public/config.js','window.AXON_API="";\n');
console.log('AXON AI website and training API ready.');
