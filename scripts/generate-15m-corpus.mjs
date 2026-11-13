#!/usr/bin/env node
// Generate a reproducible local training corpus; never commit or upload the GB file.
// Uses ~16 MB of buffer and a writable stream, not 1 GB RAM.
import fs from 'node:fs';
import path from 'node:path';
import {corpusLine} from '../server/curriculum.js';
const COUNT=15_000_001;
const MIN_SIZE=1_000_000_000;
const out=path.resolve(process.argv[2]||'AXONAI_TRAINING_15M.txt');
if(fs.existsSync(out)&&!process.argv.includes('--overwrite')){
 console.error('File already exists. Provide a new output path or --overwrite.');process.exit(2);
}
const output=fs.createWriteStream(out,{flags:'w',highWaterMark:16*1024*1024});
let count=0,bytes=0;
const write=async line=>{
 const data=Buffer.from(line,'utf8');
 if(!output.write(data))await new Promise((resolve,reject)=>{output.once('drain',resolve);output.once('error',reject)});
 bytes+=data.length;
};
try{
 for(let i=0;i<COUNT;i++){
   await write(corpusLine(i));
   count++;
   if(count%1_000_000===0)console.log('Generated '+count.toLocaleString('en-US')+' examples');
 }
 // 15M lines of this curriculum generally exceed 1GB; only extend if needed.
 for(let i=COUNT;bytes<MIN_SIZE;i++){await write(corpusLine(i));count++;}
 output.end();await new Promise((resolve,reject)=>{output.once('finish',resolve);output.once('error',reject)});
 console.log('AXON corpus saved:',out,'lines:',count,'bytes:',bytes);
 console.log('Each index maps to one deterministic example; this is synthetic training data, not pretrained intelligence.');
}catch(e){output.destroy();console.error('Corpus generation failed:',e.message);process.exitCode=1;}
