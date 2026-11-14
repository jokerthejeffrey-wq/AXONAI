// Training jobs and approved model checkpoints live on AXON AI's Vercel site.
// CPU/GPU clients do the training. Chat inference runs in the website function.
import {quality} from './train-store.js';
import {scratchAnswer} from './scratch-chat.js';
import {assistedAnswer} from './assisted-chat.js';
export async function getQuality(){try{return await quality()}catch{return null}}
export async function inferFromStudent(messages,quality){
  // When available, a separately labelled pretrained model gives useful chat
  // while the independently trained AXON student is evaluated. No chat data
  // is fed to the trainer, and failed gateway calls never expose random tokens.
  if(process.env.AXON_CHAT_MODE!=='student'){
    const reply=await assistedAnswer(messages);
    if(reply)return {reply,mode:'assisted',model:'Qwen3 via Vercel AI Gateway',version:quality?.version??0};
  }
  return scratchAnswer(messages,quality);
}
