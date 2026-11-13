import {claim} from '../server/train-store.js';
import {noCache,method,onlyJson,sendError} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);if(!method(req,res,'POST')||!onlyJson(req,res))return;
 try{const body=req.body||{};if(!body||typeof body!=='object'||Array.isArray(body)||JSON.stringify(body).length>1000)return res.status(400).json({error:'Invalid job request'});
 const job=await claim(body.worker);
 if(job===null){res.setHeader('X-AXON-Queue','review-pending');return res.status(204).end();}
 return res.status(200).json(job);}catch(e){return sendError(res,e)} }
