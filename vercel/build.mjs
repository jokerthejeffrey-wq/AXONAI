import fs from 'node:fs';
import path from 'node:path';
const raw=process.env.AXON_API_URL?.trim();
if(!raw)throw Error('Set AXON_API_URL to your public HTTPS Python backend URL in Vercel settings.');
const url=new URL(raw);
if(url.protocol!=='https:'||url.username||url.password||url.search||url.hash||url.pathname!=='/')throw Error('AXON_API_URL must be a plain HTTPS origin, e.g. https://api.example.com');
const target=path.join(import.meta.dirname,'public','config.js');
fs.writeFileSync(target,'window.AXON_API='+JSON.stringify(url.origin)+';\n');
console.log('Static frontend configured for',url.origin);
