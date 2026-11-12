import {MODEL_LABEL} from '../server/config.js';
export default function handler(req,res){
 res.setHeader('Cache-Control','no-store');
 if(req.method!=='GET')return res.status(405).json({error:'Use GET.'});
 return res.status(200).json({ok:true,chat_mode:'vercel-local',model:MODEL_LABEL,hosting:'vercel',training_available:false});
}
