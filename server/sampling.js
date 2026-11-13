// Sample model probabilities; do not always choose the same highest logit.
export function sampleToken(logits,recent=[],{temperature=.75,topK=40,topP=.9,repetitionPenalty=1.12,random=Math.random}={}){
 if(!(temperature>0&&temperature<=2&&topP>0&&topP<=1&&Number.isInteger(topK)&&topK>0))throw Error('Invalid sampling settings');
 const seen=new Set(recent.slice(-64));
 const scores=Array.from(logits,(raw,id)=>{
  let value=seen.has(id)?(raw<0?raw*repetitionPenalty:raw/repetitionPenalty):raw;
  if(id===256||id===257||id===258||!Number.isFinite(value))value=-Infinity;
  return {id,value:value/temperature};
 }).filter(x=>Number.isFinite(x.value)).sort((a,b)=>b.value-a.value).slice(0,topK);
 if(!scores.length)throw Error('No valid token probabilities');
 const max=scores[0].value;let sum=0;
 for(const s of scores){s.p=Math.exp(s.value-max);sum+=s.p;}
 let mass=0;const kept=[];
 for(const s of scores){s.p/=sum;kept.push(s);mass+=s.p;if(mass>=topP)break;}
 let pick=random()*mass;
 for(const s of kept){pick-=s.p;if(pick<=0)return s.id;}
 return kept.at(-1).id;
}
