window.axonUrl=path=>{
 const configured=window.AXON_API;
 if(configured===null||configured===undefined)throw Error('Backend is not configured. Set AXON_API_URL in Vercel and redeploy.');
 return configured.replace(/\/$/,'')+path;
};
window.axonFetch=(path,options)=>fetch(axonUrl(path),options);
document.addEventListener('DOMContentLoaded',()=>{
 const commands=document.getElementById('commands');
 if(commands){try{const base=window.AXON_API||location.origin;commands.textContent='python CPU.py --server '+base+'\n# OR\npython GPU.py --server '+base}catch(e){commands.textContent=e.message}}
});
