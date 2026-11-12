import os,argparse,tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests
from settings import manifest

def run(device):
    p=argparse.ArgumentParser();p.add_argument('--server',default=os.getenv('AXON_SERVER',''))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--rounds',type=int,default=1);p.add_argument('--yes',action='store_true')
    a=p.parse_args();url=(a.server or input('Training backend URL (not the Vercel frontend): ').strip()).rstrip('/')
    parts=urlparse(url)
    if parts.scheme!='https' and not(parts.scheme=='http' and parts.hostname in ('localhost','127.0.0.1')):p.error('Remote backend requires HTTPS.')
    if not 1<=a.threads<=64 or not 1<=a.rounds<=100:p.error('threads 1..64, rounds 1..100')
    print('AXONAI from-scratch student training |',device)
    print('Downloads student checkpoints and teacher-written lessons. Only the owner runs the teacher.')
    print('No personal files are read. Uses compute and electricity. Ctrl+C stops.')
    if not a.yes and input('Start training and upload results? [y/N] ').lower()!='y':return
    import torch
    from student import Student,load_weights,train,snapshot
    from safetensors.torch import save_file
    torch.set_num_threads(a.threads)
    if device=='cuda' and not torch.cuda.is_available():
        print('CUDA unavailable. A compatible NVIDIA driver is required; no driver changes were attempted.')
        if input('Use CPU for this session instead? [y/N] ').strip().lower()!='y':raise SystemExit('Stopped. You can also run START_CPU.cmd.')
        device='cpu'
    model=Student().to(device);session=requests.Session()
    if os.getenv('AXON_TRAIN_TOKEN'):session.headers['Authorization']='Bearer '+os.environ['AXON_TRAIN_TOKEN']
    for number in range(a.rounds):
        r=session.post(url+'/api/job',timeout=60)
        if r.status_code==204:print('No unused lesson budget remains. Owner must add NEW topics.');break
        if r.status_code==503:print(r.json().get('error','Owner setup needed'));break
        r.raise_for_status();job=r.json()
        if job['manifest']!=manifest():raise RuntimeError('Wrong client protocol. Download the current client ZIP.')
        with tempfile.TemporaryDirectory(prefix='axon-') as folder:
            checkpoint=Path(folder)/'student.safetensors'
            with session.get(url+f"/api/checkpoint/{job['version']}",stream=True,timeout=120) as download:
                download.raise_for_status();size=0
                with checkpoint.open('wb') as f:
                    for block in download.iter_content(1024*1024):
                        size+=len(block)
                        if size>32*1024*1024:raise ValueError('Checkpoint exceeds protocol limit')
                        f.write(block)
            load_weights(model,checkpoint)
            print(f"Round {number+1}: {len(job['rows'])} lessons, one step per lesson; base v{job['version']}")
            try:train(model,job['rows'],device)
            except torch.OutOfMemoryError:
                raise SystemExit('Out of memory. This job will not be replayed. Close other apps or use CPU with more RAM.')
            save_file(snapshot(model),str(checkpoint))
            with checkpoint.open('rb') as f:r=session.post(url+'/api/update',data={'job':job['id']},files={'weights':('student.safetensors',f)},timeout=180)
            if r.status_code==409:print('Update became stale. Reserved uses remain consumed; no lesson was replayed.');continue
            r.raise_for_status();print('Submitted for owner review:',r.json()['id'])
    print('Finished; no background process remains.')
