window.axonUrl=path=>path;
window.axonFetch=async(path,options={})=>{
 let response;
 try{response=await fetch(path,{...options,signal:AbortSignal.timeout(295000)})}
 catch(error){throw Error(error.name==='TimeoutError'?'The model took too long to start. Please retry shortly.':'Cannot reach this website’s API. Check your connection and reload the page.');}
 const type=response.headers.get('content-type')||'';
 if(!type.includes('application/json'))throw Error(response.status===504?'The cloud model timed out. Please retry; its files may now be cached.':'The Vercel API is missing or unavailable (HTTP '+response.status+'). Redeploy the complete project, including api/.');
 return response;
};
