const cells=document.getElementById('cells');for(let i=0;i<216;i++){const c=document.createElement('i');if(i*7%37<3)c.className='lit';cells.append(c)}
async function status(){try{const r=await axonFetch('/api/status');if(!r.ok)throw Error();document.getElementById('training-status').textContent='Chat on Vercel · training external'}catch(e){document.getElementById('training-status').textContent='Chat API unavailable · training external'}}
status();setInterval(status,30000);

// Restored from the original AXONAI landing page.
let effectsOn=!matchMedia("(prefers-reduced-motion: reduce)").matches;
// Slow, time-driven wave. Pointer position is never used.
(function(){
const c=document.getElementById('gridwave'),ctx=c.getContext('2d');let w=0,h=0,phase=0,last=0,visible=true;
function size(){w=c.clientWidth;h=c.clientHeight;const d=Math.min(devicePixelRatio||1,2);c.width=w*d;c.height=h*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function draw(){ctx.clearRect(0,0,w,h);
const P=(x,z)=>{const k=600/(600+z*13);return [w/2+x*(w/34)*k,h*.76+(z*13-Math.sin(x*.3+z*.24+phase)*28-Math.cos(x*.17-z*.3+phase*.65)*14)*k]};
for(let z=-8;z<24;z++){ctx.beginPath();for(let x=-27;x<=27;x++){const p=P(x,z);x===-27?ctx.moveTo(...p):ctx.lineTo(...p)}ctx.strokeStyle='rgba(255,255,255,'+(.25*(1-z/35))+')';ctx.lineWidth=.65;ctx.stroke()}
for(let x=-27;x<=27;x++){ctx.beginPath();for(let z=-8;z<=24;z++){const p=P(x,z);z===-8?ctx.moveTo(...p):ctx.lineTo(...p)}ctx.strokeStyle='rgba(255,255,255,.19)';ctx.stroke()}}
function frame(now){const dt=last?Math.min((now-last)/1000,.06):0;last=now;if(effectsOn&&visible&&!document.hidden){phase+=dt*.14;draw()}requestAnimationFrame(frame)}
new IntersectionObserver(e=>{visible=e[0].isIntersecting}).observe(c);size();addEventListener('resize',size);requestAnimationFrame(frame);
})();

matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", e=>{effectsOn=!e.matches});
