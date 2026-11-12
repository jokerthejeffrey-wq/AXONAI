import {uploadPart} from '../server/train-store.js';
import {CHUNK_BYTES} from '../server/train-core.js';
import {noCache,method,binaryBody,textHeader,sendError} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);if(!method(req,res,'POST'))return;
 try{
   if(textHeader(req,'content-type')!=='application/octet-stream')return res.status(415).json({error:'Use application/octet-stream'});
   const part=Number(textHeader(req,'x-axon-part'));
   const data=await binaryBody(req,CHUNK_BYTES);
   return res.status(200).json(await uploadPart(textHeader(req,'x-axon-job'),textHeader(req,'x-axon-token'),part,data));
 }catch(e){return sendError(res,e)} }
export const config={api:{bodyParser:false}};
