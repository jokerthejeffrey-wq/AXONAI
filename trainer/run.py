"""Vercel/Neon/Blob trainer: same student architecture, local corpus and teacher."""
import argparse,os,sys,json,hashlib,time,uuid,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from trainer.corpus import prepare,read_rows
from trainer.memory import wait

def request(session,method,url,**kwargs):
    for attempt in range(10):
        r=session.request(method,url,timeout=300,**kwargs)
        if r.status_code not in (429,503):r.raise_for_status();return r
        print('Server busy; retrying in 30 seconds.',flush=True);time.sleep(30)
    raise RuntimeError('Server remains busy. Restart later; completed reservations will not be reassigned.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--server',default='https://axonai-theta.vercel.app');p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--cache',default=str(Path.home()/'.axonai'/'smoltalk-v1'));p.add_argument('--yes',action='store_true');args=p.parse_args()
    from urllib.parse import urlparse
    parsed=urlparse(args.server)
    if parsed.scheme!='https' or parsed.username or parsed.password:p.error('An HTTPS coordinator URL is required')
    if not 1<=args.rounds<=1000:p.error('Rounds must be 1..1000')
    print('Downloads up to 1 GB conversational TXT plus local teacher, dependencies and checkpoints. Dataset cache/index requires additional disk space. Uses your CPU/GPU and uploads student weights.')
    if not args.yes and input('Start? [y/N] ').lower()!='y':return
    wait(512);info=prepare(args.cache)
    wait(3000)
    import torch
    from safetensors.torch import save_file
    from student import Student,load_weights,snapshot,train
    from trainer.teacher import Teacher
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA driver unavailable. Start CPU edition instead.')
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.85)
    worker_file=Path(args.cache)/'worker-id'
    if not worker_file.exists():worker_file.write_text(str(uuid.uuid4()))
    worker=worker_file.read_text().strip();session=requests.Session();base=args.server.rstrip('/')
    teacher=Teacher(args.device)
    try:
        model=Student().to(args.device)
        for n in range(args.rounds):
            wait(512)
            r=request(session,'POST',base+'/api/job',json={'worker':worker,'corpus':info})
            if r.status_code==204:print('Queue/quota or unique corpus range exhausted; stopping.');break
            job=r.json();assignment=job.get('corpus')
            if not assignment or assignment['id']!=info['id'] or assignment['revision']!=info['revision']:raise RuntimeError('Server does not support this corpus client yet.')
            arch=job['manifest']['architecture']
            if arch!=model.config:raise RuntimeError('Student architecture mismatch')
            rows=[]
            for offset,raw in enumerate(read_rows(args.cache,assignment['start'],assignment['count'])):
                print(f"Teacher preparing record {assignment['start']+offset+1}…",flush=True)
                wait(256);row=teacher.lesson(raw)
                if row:rows.append(row)
            if not rows:print('No safe short examples in this reservation; skipping without replay.');continue
            manifest=request(session,'GET',base+f"/api/checkpoint/{job['version']}").json()
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'student.safetensors';digest=hashlib.sha256();size=0
                with file.open('wb') as out:
                    for part in manifest['parts']:
                        url=part['url'];url=base+url if url.startswith('/model/') else url
                        host=urlparse(url)
                        if host.scheme!='https' or not(host.netloc==parsed.netloc or host.hostname.endswith('.public.blob.vercel-storage.com')):raise RuntimeError('Untrusted checkpoint host')
                        with session.get(url,stream=True,timeout=120) as download:
                            download.raise_for_status()
                            for block in download.iter_content(1024*1024):
                                size+=len(block)
                                if size>32*1024**2:raise RuntimeError('Oversized checkpoint')
                                out.write(block);digest.update(block)
                if size!=manifest['size'] or digest.hexdigest()!=manifest['sha256']:raise RuntimeError('Checkpoint checksum mismatch')
                load_weights(model,file)
                # Do not replay this reservation, even if upload fails after training.
                train(model,rows,args.device,before_step=lambda:wait(256))
                save_file(snapshot(model),str(file));data=file.read_bytes();chunk_size=2_500_000
                parts=(len(data)+chunk_size-1)//chunk_size
                for index in range(parts):
                    request(session,'POST',base+'/api/update-part',data=data[index*chunk_size:(index+1)*chunk_size],headers={'Content-Type':'application/octet-stream','X-AXON-Job':job['id'],'X-AXON-Token':job['upload_token'],'X-AXON-Part':str(index)})
                result=request(session,'POST',base+'/api/update',json={'job':job['id'],'token':job['upload_token'],'parts':parts,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                print(f"Round {n+1}: {len(rows)} distinct conversation examples submitted for validation.",flush=True)
    finally:teacher.close()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nStopped.')
