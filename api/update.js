import {finish} from '../server/train-store.js';
import {noCache,method,onlyJson,sendError} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);if(!method(req,res,'POST')||!onlyJson(req,res))return;
 try{const b=req.body||{};return res.status(200).json(await finish(b.job,b.token,b.parts,b.sha256,b.size));}catch(e){return sendError(res,e)} }
