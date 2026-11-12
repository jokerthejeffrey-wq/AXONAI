import os from 'node:os';
import {readFileSync} from 'node:fs';

function positiveNumber(value){const n=Number(String(value).trim());return Number.isFinite(n)&&n>0?n:null;}
export function memoryStats(read=readFileSync){
  // Linux cgroup limits are often more accurate than host RAM for Vercel functions.
  const paths=[['/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.max'],['/sys/fs/cgroup/memory/memory.usage_in_bytes','/sys/fs/cgroup/memory/memory.limit_in_bytes']];
  for(const [usedFile,maxFile] of paths){
    try{
      const used=positiveNumber(read(usedFile,'utf8'));
      const limit=positiveNumber(read(maxFile,'utf8'));
      if(used&&limit&&limit<2**52)return {used,total:limit,source:'cgroup'};
    }catch{}
  }
  return {used:os.totalmem()-os.freemem(),total:os.totalmem(),source:'system'};
}
export function memoryAdmit(stats=memoryStats(),maxFraction=.9,reserveBytes=128*1024*1024){
  return stats.total>0&&(stats.used+reserveBytes)/stats.total<=maxFraction;
}
