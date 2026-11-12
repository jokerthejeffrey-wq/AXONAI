import fs from 'node:fs';
fs.writeFileSync('public/config.js','window.AXON_API=""; // Same-origin Vercel functions. No PC or tunnel.\n');
console.log('AXONAI: same-origin Node inference enabled; no AXON_API_URL required.');
