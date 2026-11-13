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
export function localBasics(text){
  const t=String(text).trim().toLowerCase().replace(/[.!?]+$/,'');
  if(/^(hello|hi|hey|hello axon|hi axon|good morning|good evening|good afternoon)$/.test(t))
    return 'Hello! What would you like to ask?';
  if(/^(what can you help me with|what can you do|help)$/.test(t))
    return 'I can handle basic calculations and explain the AXON AI project. My independently trained language model is still learning.';
  if(/^(explain how ai learns|how does ai learn|how do ai models learn)$/.test(t))
    return 'An AI model learns by adjusting its numerical weights using examples. AXON AI volunteers train its model on their computers, then submit updates to the website for review.';
  if(/^(help me start a project|how do i start a project)$/.test(t))
    return 'Start by describing your goal, choosing one small feature, and testing it. AXON AI is still learning, so I may not be able to guide every step.';
  return null;
}
export const STILL_TRAINING='AXON AI is still learning. I cannot answer that reliably yet.';
export async function scratchAnswer(messages,quality,{load=loadStudent,generate=scratchGenerate}={}){
  const version=Number(quality?.version||0),basic=localBasics(messages.at(-1).content);
  if(basic&&version===0)return {reply:basic,mode:'local-basic',version};
  if(!Number.isSafeInteger(version)||version<0)return {reply:STILL_TRAINING,mode:'learning',version:0};
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
