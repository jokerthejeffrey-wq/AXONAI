import {getQuality} from '../server/coordinator.js';
import {publicStatus} from '../server/train-store.js';
import {parameterCount,LEGACY_ARCHITECTURE,EXPANDED_ARCHITECTURE,COMPACT_ARCHITECTURE} from '../server/model-profiles.js';
export default async function handler(req,res){
  res.setHeader('Cache-Control','no-store');
  if(req.method!=='GET')return res.status(405).json({error:'Use GET.'});
  const [quality,training]=await Promise.all([getQuality(),publicStatus()]);
  return res.status(200).json({
    ok:true,model:'AXON AI',hosting:'vercel',
    chat_mode:'pretrained-generation-with-browser-fallback',
    chat_requires_gateway:false,local_model_installed:true,
    chat_browser_model:'SmolLM2 360M (CPU) / Qwen2.5 0.5B (WebGPU)',
    chat_student_ready:Boolean(quality?.ready&&quality?.tested),
    training_available:Boolean(training.training_available),
    training_status_reason:training.reason||null,
    training_progress:training.reserved_lessons??0,
    training_version:training.version??0,pending_updates:training.pending_updates??0,
    active_model_parameters:parameterCount(LEGACY_ARCHITECTURE),
    experimental_model_parameters:parameterCount(EXPANDED_ARCHITECTURE),
    experimental_model_deployed:false,
    compact_research_parameters:parameterCount(COMPACT_ARCHITECTURE),
    compact_research_deployed:false,
    distinct_training_contributors_7d:training.contributing_workers_7d??0,
    five_contributors_active:(training.contributing_workers_7d??0)>=5,
    quality:quality?{tested:Boolean(quality.tested),score:quality.score??null,
      ready:Boolean(quality.ready),version:quality.version??0,
      passed:quality.passed??0,total:quality.total??0,
      checked_at:quality.checked_at??null,review:quality.review||'pending'}:null
  });
}
