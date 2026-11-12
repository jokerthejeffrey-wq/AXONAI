import {TrainError} from './train-store.js';
export function sendError(res,error){
  const status=error instanceof TrainError?error.status:503;
  if(status===429||status===503)res.setHeader('Retry-After',status===429?'120':'30');
  if(!(error instanceof TrainError))console.error('AXON training request failed:',error?.name);
  return res.status(status).json({error:error instanceof TrainError?error.message:'Training service unavailable.'});
}
export function noCache(res){res.setHeader('Cache-Control','no-store');}
export function method(req,res,verb){if(req.method!==verb){res.setHeader('Allow',verb);res.status(405).json({error:'Use '+verb});return false;}return true;}
export function textHeader(req,key){const v=req.headers[key.toLowerCase()];return typeof v==='string'?v:''}
export async function binaryBody(req,max){
  const declared=Number(textHeader(req,'content-length'));
  if(!Number.isSafeInteger(declared)||declared<=0||declared>max)throw new TrainError('Chunk too large or empty',413);
  if(Buffer.isBuffer(req.body)){if(req.body.length>max)throw new TrainError('Chunk too large',413);return req.body;}
  if(req.body instanceof Uint8Array){return Buffer.from(req.body);}
  const chunks=[];let length=0;
  for await(const chunk of req){length+=chunk.length;if(length>max)throw new TrainError('Chunk too large',413);chunks.push(chunk);}
  return Buffer.concat(chunks);
}
export function onlyJson(req,res){if(!String(textHeader(req,'content-type')).startsWith('application/json')){res.status(415).json({error:'Use application/json'});return false;}return true}
