(()=>{
 const root=document.documentElement;
 let theme='light';
 try{theme=localStorage.getItem('axon-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}catch(e){}
 root.dataset.theme=theme==='dark'?'dark':'light';
 const sun='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="4"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2"/></svg>';
 const moon='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M20.5 14.5A9 9 0 0 1 9.5 3.5 9 9 0 1 0 20.5 14.5Z"/></svg>';
 let timer;
 function labels(){document.querySelectorAll('.theme-toggle').forEach(b=>{
  b.setAttribute('aria-label',root.dataset.theme==='dark'?'Switch to light theme':'Switch to dark theme');
  b.title=b.getAttribute('aria-label');
 })}
 document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.theme-toggle').forEach(b=>{
   b.innerHTML='<span class="theme-sun" aria-hidden="true">'+sun+'</span><span class="theme-moon" aria-hidden="true">'+moon+'</span>';
   b.onclick=()=>{
    clearTimeout(timer);
    if(!matchMedia('(prefers-reduced-motion: reduce)').matches)root.classList.add('theme-changing');
    root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';
    try{localStorage.setItem('axon-theme',root.dataset.theme)}catch(e){}
    labels();timer=setTimeout(()=>root.classList.remove('theme-changing'),450);
   };
  });labels();
 });
})();
