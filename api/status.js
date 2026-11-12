import { getQuality } from '../server/coordinator.js';
import { gatewayToken } from '../server/gateway.js';
export default async function handler(req,res) {
  res.setHeader('Cache-Control','no-store');
  if (req.method!=='GET') return res.status(405).json({error:'Use GET.'});
  const quality=await getQuality();
  return res.status(200).json({
    ok:true,
    model:'AXON AI',
    hosting:'vercel',
    chat_mode:quality?.ready?'student-eligible':'pretrained',
    // This checks that credentials are present, not that billing or the provider is healthy.
    chat_credentials_present:Boolean(gatewayToken(req)),
    local_model_installed:false,
    training_available:Boolean(quality),
    quality:quality?{
      tested:Boolean(quality.tested),score:quality.score??null,
      ready:Boolean(quality.ready),version:quality.version??0,
      passed:quality.passed??0,total:quality.total??0,
      checked_at:quality.checked_at??null,review:quality.review||'pending'
    }:null
  });
}
