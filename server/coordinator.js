// The AXON training API lives on this Vercel website, not a separate machine.
// Student inference stays disabled until an independently validated model
// deployment exists; the pretrained chat must not silently switch to untested weights.
import {quality} from './train-store.js';
export async function getQuality(){try{return await quality()}catch{return null}}
export async function inferFromStudent(){return null}
