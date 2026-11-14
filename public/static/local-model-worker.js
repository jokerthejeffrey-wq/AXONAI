// Genuine subword-token inference, using published open pretrained weights.
// Runs in a dedicated worker. Does not train or modify AXON's student weights.
const LIBRARY='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';
const SMALL='onnx-community/SmolLM2-360M-Instruct-ONNX';
const LARGE='onnx-community/Qwen2.5-0.5B-Instruct';
let engine=null,loadedModel=null,active=false;
function send(type,extra={}){self.postMessage({type,...extra});}
function onProgress(p){
  if(p.status==='progress'&&Number.isFinite(Number(p.progress)))
    send('status',{text:'Downloading pretrained model: '+Math.max(0,Math.min(100,Math.round(Number(p.progress))))+'%'});
  else if(p.status==='initiate')send('status',{text:'Loading '+(p.file||'model files')+'…'});
}
async function init(){
  if(engine)return engine;
  send('status',{text:'Initializing pretrained language model (first use downloads hundreds of MB)…'});
  const {pipeline,TextStreamer,env}=await import(LIBRARY);
  env.allowLocalModels=false;
  env.backends.onnx.wasm.numThreads=1;
  let gpu=false;
  try{gpu=Boolean(self.navigator?.gpu&&await self.navigator.gpu.requestAdapter());}catch{}
  if(gpu){
    try{
      const generator=await pipeline('text-generation',LARGE,{
        dtype:'q4f16',device:'webgpu',progress_callback:onProgress
      });
      engine={generator,TextStreamer};
      loadedModel='Qwen2.5 0.5B (WebGPU pretrained)';
      return engine;
    }catch(error){
      send('status',{text:'GPU model unavailable; loading CPU-compatible pretrained model…'});
    }
  }
  const generator=await pipeline('text-generation',SMALL,{
    dtype:'q4',device:'wasm',progress_callback:onProgress
  });
  engine={generator,TextStreamer};
  loadedModel='SmolLM2 360M (local pretrained)';
  return engine;
}
self.onmessage=async({data})=>{
  if(data?.type!=='generate'||active)return;
  active=true;let reply='';
  try{
    if(!Array.isArray(data.messages)||!data.messages.length)throw Error('Missing conversation');
    const {generator,TextStreamer}=await init();
    send('meta',{mode:'local-pretrained',model:loadedModel});
    send('status',{text:'Generating with '+loadedModel+'…'});
    const streamer=new TextStreamer(generator.tokenizer,{
      skip_prompt:true,skip_special_tokens:true,
      callback_function:part=>{
        if(typeof part==='string'&&part){
          reply+=part;
          send('delta',{text:part});
        }
      }
    });
    const history=data.messages.slice(-9).filter(m=>['user','assistant'].includes(m.role)&&typeof m.content==='string');
    const input=[
      {role:'system',content:'You are AXON AI, a helpful conversational assistant. Respond directly, naturally and briefly. You are running pretrained inference, not the experimental AXON student model.'},
      ...history
    ];
    await generator(input,{max_new_tokens:110,do_sample:false,repetition_penalty:1.08,streamer});
    if(!reply.trim())throw Error('The language model produced no reply.');
    send('done',{reply:reply.trim(),mode:'local-pretrained',model:loadedModel});
  }catch(error){
    console.error('AXON browser model error:',String(error?.name||'Error'));
    send('error',{error:'Real language model could not run in this browser. Check that the model download is allowed and that enough RAM is available.'});
  }finally{active=false;}
};
