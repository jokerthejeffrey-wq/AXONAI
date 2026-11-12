import {review,reviewList} from '../server/train-store.js';
import {noCache,sendError,textHeader} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);
 const value=textHeader(req,'authorization');const token=value.startsWith('Bearer ')?value.slice(7):'';
 try{
   if(req.method==='GET')return res.status(200).json(await reviewList(token));
   if(req.method==='POST'){const b=req.body||{};return res.status(200).json(await review(token,b.id,b.action));}
   return res.status(405).json({error:'Method not allowed'});
 }catch(e){return sendError(res,e)} }
