// Self-hosted AXON AI: inference uses only checkpoint weights saved on this site.
import {manifestFor,assemble} from './train-store.js';
import {sha256} from './train-core.js';
import {parseCheckpoint,scratchGenerate} from './scratch-infer.js';
let loaded=null;
export async function loadStudent(version){
  if(loaded?.version===version)return loaded.promise;
  const promise=(async()=>{
    const manifest=await manifestFor(version);
    const bytes=await assemble(manifest.parts);
    if(bytes.length!==manifest.size||sha256(bytes)!==manifest.sha256)throw Error('Checkpoint integrity check failed');
    return parseCheckpoint(bytes);
  })();
  loaded={version,promise};
  try{return await promise}catch(error){if(loaded?.promise===promise)loaded=null;throw error;}
}
export function readableReply(answer){
  if(typeof answer!=='string'||answer.length<12||answer.length>500)return false;
  if(/[\u0000-\u001f\u007f\ufffd]/.test(answer)||/(.)\1{5,}/u.test(answer))return false;
  const chars=[...answer],letters=chars.filter(c=>/\p{L}/u.test(c));
  return letters.length>=7&&letters.length/chars.length>=.6&&
    new Set(letters.map(c=>c.toLowerCase())).size>=5&&answer.trim().split(/\s+/u).length>=3;
}
export const STILL_TRAINING='AXON AI is still learning. I cannot answer that reliably yet.';
export async function scratchAnswer(messages,quality,{load=loadStudent,generate=scratchGenerate}={}){
  const version=Number(quality?.version||0);
  if(!Number.isSafeInteger(version)||version<0)return {reply:STILL_TRAINING,mode:'learning',version:0};
  // A newly saved checkpoint is NOT a chat-quality pass. Never stream or show
  // its raw token samples until a held-out evaluation explicitly marks it ready.
  if(quality?.tested!==true || quality?.ready!==true)
    return {reply:STILL_TRAINING,mode:'learning',version};
  try{
    const model=await load(version);
    const answer=generate(model,messages,{maxNewTokens:128});
    // Version 0 contains random initial weights and cannot be trusted.
    if(version>0&&readableReply(answer))return {reply:answer,mode:'student',version};
  }catch(error){
    console.error('AXON scratch model:',String(error?.name||'Error').slice(0,32));
  }
  return {reply:STILL_TRAINING,mode:'learning',version};
}
