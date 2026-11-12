import {manifestFor} from '../server/train-store.js';
import {noCache,method,sendError} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);if(!method(req,res,'GET'))return;
 try{const raw=req.query?.version;const version=Number(raw);if(!/^\d+$/.test(String(raw))||!Number.isSafeInteger(version))return res.status(400).json({error:'Invalid version'});
 return res.status(200).json(await manifestFor(version));}catch(e){return sendError(res,e)} }
