import {submissionState} from '../server/train-store.js';
import {noCache,method,onlyJson,sendError} from '../server/train-http.js';
export default async function handler(req,res){
  noCache(res);
  if(!method(req,res,'POST')||!onlyJson(req,res))return;
  try{
    const body=req.body||{};
    if(typeof body.job!=='string'||typeof body.token!=='string')
      return res.status(400).json({error:'Invalid job credentials'});
    return res.status(200).json(await submissionState(body.job,body.token));
  }catch(error){return sendError(res,error)}
}
