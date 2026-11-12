import os from 'node:os';
import path from 'node:path';
import {MODEL_ID,MODEL_REVISION,SYSTEM} from './config.js';
let loading;
export async function getGenerator(){
 if(!loading){
  loading=(async()=>{
   const {pipeline,env}=await import('@huggingface/transformers');
   // The deployment is read-only. Only the server's temporary cache is writable.
   env.cacheDir=path.join(process.env.AXON_CACHE_DIR||os.tmpdir(),'axon-smollm2');
   env.allowLocalModels=false;
   return pipeline('text-generation',MODEL_ID,{
    revision:MODEL_REVISION,dtype:'q4',device:'cpu',
    session_options:{intraOpNumThreads:1,interOpNumThreads:1},
   });
  })().catch(error=>{loading=undefined;throw error;});
 }
 return loading;
}
export async function generate(messages){
 const generator=await getGenerator();
 let context=[{role:'system',content:SYSTEM},...messages.slice(-5)];
 // Keep chat roles valid when trimming old pairs.
 if(context[1]?.role==='assistant')context.splice(1,1);
 while(true){
  const tokens=generator.tokenizer.apply_chat_template(context,{tokenize:true,add_generation_prompt:true});
  const size=tokens.length??tokens.input_ids?.size??0;
  if(size<=768)break;
  if(context.length>2)context.splice(1,2);
  else throw Error('Your message is too long for this small model. Please shorten it.');
 }
 const output=await generator(context,{
  max_new_tokens:128,do_sample:true,temperature:.65,top_p:.9,repetition_penalty:1.1,
 });
 const generated=output[0]?.generated_text;
 const reply=Array.isArray(generated)?generated.at(-1)?.content:generated;
 if(typeof reply!=='string'||!reply.trim())throw Error('The model returned an empty reply. Please retry.');
 return reply.trim();
}
