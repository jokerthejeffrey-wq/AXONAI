import {getQuality} from '../server/coordinator.js';
import {publicStatus} from '../server/train-store.js';
export default async function handler(req,res){
  res.setHeader('Cache-Control','no-store');
  if(req.method!=='GET')return res.status(405).json({error:'Use GET.'});
  const [quality,training]=await Promise.all([getQuality(),publicStatus()]);
  return res.status(200).json({
    ok:true,model:'AXON AI',hosting:'vercel',chat_mode:'local-scratch',
    chat_requires_gateway:false,local_model_installed:true,
    training_available:Boolean(training.training_available),
    training_progress:training.reserved_lessons??0,
    training_version:training.version??0,pending_updates:training.pending_updates??0,
    quality:quality?{tested:Boolean(quality.tested),score:quality.score??null,
      ready:Boolean(quality.ready),version:quality.version??0,
      passed:quality.passed??0,total:quality.total??0,
      checked_at:quality.checked_at??null,review:quality.review||'pending'}:null
  });
}
