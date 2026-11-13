import {importTeacherRows} from '../server/train-store.js';
import {noCache,method,onlyJson,sendError,textHeader} from '../server/train-http.js';
export default async function handler(req,res){
  noCache(res);
  if(!method(req,res,'POST')||!onlyJson(req,res))return;
  const token=textHeader(req,'authorization');
  const owner=token.startsWith('Bearer ')?token.slice(7):'';
  const size=Number(textHeader(req,'content-length')||0);
  if(size>60_000)return res.status(413).json({error:'Request too large'});
  try{
    const body=req.body||{};
    return res.status(200).json(await importTeacherRows(owner,body.rows));
  }catch(error){return sendError(res,error)}
}
