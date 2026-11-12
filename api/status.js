import {MODEL_LABEL} from '../server/config.js';
import {getQuality} from '../server/coordinator.js';
export default async function handler(req,res){
 res.setHeader('Cache-Control','no-store');
 if(req.method!=='GET')return res.status(405).json({error:'Use GET.'});
 const quality=await getQuality();
 return res.status(200).json({ok:true,chat_mode:quality?.ready?'student-eligible':'pretrained',
   model:MODEL_LABEL,hosting:'vercel',training_available:Boolean(quality),
   quality:quality?{tested:Boolean(quality.tested),score:quality.score??null,ready:Boolean(quality.ready),version:quality.version??0,
    passed:quality.passed??0,total:quality.total??0,checked_at:quality.checked_at??null,review:quality.review||'pending'}:null});
}
