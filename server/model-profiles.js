// Describes future AXON checkpoints; deliberately independent of the
// active protocol v2 manifest until the 50M storage track is tested.
export const LEGACY_ARCHITECTURE = Object.freeze({
  vocab_size:260,width:256,layers:4,heads:8,context:512
});
export const EXPANDED_ARCHITECTURE = Object.freeze({
  vocab_size:260,width:640,layers:10,heads:8,context:1024
});
export function parameterCount(profile){
  for(const k of ['vocab_size','width','layers','heads','context']){
    if(!Number.isSafeInteger(profile?.[k])||profile[k]<=0)throw new TypeError('Invalid model profile');
  }
  if(profile.width%profile.heads!==0)throw new TypeError('Attention width/head mismatch');
  const {vocab_size:v,width:d,layers:l,context:c}=profile;
  return l*(12*d*d+13*d)+2*v*d+c*d+2*d;
}
export function checkpointBytes(profile){
  // Includes float32 weights only; safetensors header adds a few KB.
  return parameterCount(profile)*4;
}
