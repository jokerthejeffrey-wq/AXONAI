const base=window.AXON_RELEASE_ASSET_BASE;
if(typeof base==='string' && /^https:\/\//i.test(base)){
  for(const [mode,name] of [['gpu','GPU-AXON.exe'],['cpu','CPU-AXON.exe']]){
    const link=document.getElementById(mode+'-download');
    link.href=base.replace(/\/$/,'')+'/'+name;
    link.hidden=false;
  }
  document.querySelector('.warn').hidden=true;
}
