// AXON AI's reproducible training corpus. Generated locally, never in a
// Vercel Function. The EXEs already obtain the same indexed lessons from
// the training API without needing to download this entire file.
import {createWriteStream,existsSync} from 'node:fs';
import {once} from 'node:events';
import {lesson} from '../server/train-core.js';

const output=process.argv[2]||'AXON-Training-Data.txt';
const sample=process.argv[3]==='--sample' ? 2500 : Number(process.argv[3]||0);
if(!Number.isSafeInteger(sample)||sample<0||sample>100_000)throw Error('Sample size must be 1-100000');
if(existsSync(output))throw Error('Output already exists: '+output);
const full=sample===0;
const minimumLines=full?15_000_001:sample;
const minimumBytes=full?1_073_741_824:0;
const maxLines=full?30_000_000:sample;
const outputStream=createWriteStream(output,{flags:'wx',highWaterMark:1024*1024});
let written=0,lines=0;
try{
 for(let i=0;i<maxLines&&(lines<minimumLines||written<minimumBytes);i++){
  const item=lesson(i);
  const prompt=item.messages[0].content.replace(/[\t\r\n]/g,' ');
  const answer=item.response.replace(/[\t\r\n]/g,' ');
  const line=prompt+'\t'+answer+'\n';
  const bytes=Buffer.byteLength(line);
  if(!outputStream.write(line))await once(outputStream,'drain');
  written+=bytes;lines++;
  if(lines%1_000_000===0)console.log('Examples:',lines,'Size GiB:',(written/2**30).toFixed(3));
 }
 await new Promise((resolve,reject)=>outputStream.end(error=>error?reject(error):resolve()));
} catch(error) {
 outputStream.destroy();
 throw error;
}
console.log('Created',output,'Examples:',lines,'Bytes:',written);
if(full&&(lines<15_000_001||written<minimumBytes))process.exitCode=1;
