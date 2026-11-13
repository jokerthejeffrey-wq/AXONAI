// Training jobs and approved model checkpoints live on AXON AI's Vercel site.
// CPU/GPU clients do the training. Chat inference runs in the website function.
import {quality} from './train-store.js';
import {scratchAnswer} from './scratch-chat.js';
export async function getQuality(){try{return await quality()}catch{return null}}
export async function inferFromStudent(messages,quality){return scratchAnswer(messages,quality)}
