import fs from 'node:fs';
fs.writeFileSync('public/config.js','window.AXON_API="";\n');
console.log('AXON AI build complete');
