"""Full-gradient transformer trainer with on-demand verified lesson pages."""
import argparse,os,sys,json,hashlib,time,uuid,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from trainer.chunked_corpus import load_manifest, read_assigned
from trainer.memory import wait, MemoryAdmissionError

def request(session,method,url,**kwargs):
    for attempt in range(10):
        r=session.request(method,url,timeout=300,**kwargs)
        if r.status_code not in (429,503):r.raise_for_status();return r
        print('Server busy; retrying in 30 seconds.',flush=True);time.sleep(30)
    raise RuntimeError('Server remains busy. Restart later; completed reservations will not be reassigned.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--server',default='https://axonai-theta.vercel.app');p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--teacher', action='store_true', help='Optional local pretrained reviewer; adds download and CPU time')
    p.add_argument('--cache',default=str(Path.home()/'.axonai'/'smoltalk-v1'));p.add_argument('--yes',action='store_true');p.add_argument('--no-teacher',action='store_true');args=p.parse_args()
    from urllib.parse import urlparse
    parsed=urlparse(args.server)
    if parsed.scheme!='https' or parsed.username or parsed.password:p.error('An HTTPS coordinator URL is required')
    if not 1<=args.rounds<=1000:p.error('Rounds must be 1..1000')
    base=args.server.rstrip('/')
    try:
        response=requests.get(base+'/api/status',timeout=(8,15))
        response.raise_for_status()
        state=response.json()
        if state.get('training_available') is not True:
            raise RuntimeError('Training coordinator not ready: '+str(state.get('training_status_reason') or 'storage unavailable'))
    except requests.RequestException as error:
        raise RuntimeError('Training preflight failed. Verify the coordinator URL and internet connection.') from error
    print('Training: unique conversational corpus and full-model weight updates.')
    print('Pretrained review is optional; it is never mixed into student weights.', flush=True)
    print('Downloads are cached locally. Press Ctrl+C to stop.', flush=True)
    pages=load_manifest();info=pages['corpus']
    Path(args.cache).mkdir(parents=True,exist_ok=True)
    print('Fast setup: only assigned lesson pages are downloaded; no 1 GB corpus build.', flush=True)
    wait(512)
    import torch
    from student import Student, load_weights, snapshot, train as train_model
    from safetensors.torch import save_file
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Launch CPU-AXON.exe instead.')
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    print('Training full AXON transformer on '+args.device.upper()+'.', flush=True)
    # Do not load a large pretrained model before the server allocates work.
    # If it cannot load on a low-memory computer, the base trainer still works.
    teacher=None
    teacher_attempted=False
    worker_file=Path(args.cache)/'worker-id'
    if not worker_file.exists():worker_file.write_text(str(uuid.uuid4()))
    worker=worker_file.read_text().strip();session=requests.Session()
    try:
        model=Student().to(args.device)
        for n in range(args.rounds):
            wait(512)
            r=request(session,'POST',base+'/api/job',json={'worker':worker,'corpus':info})
            if r.status_code==204:
                reason=r.headers.get('X-AXON-Queue','review-pending')
                details={
                    'worker-daily-limit':'Daily training share complete.',
                    'daily-capacity-reached':'Training capacity filled for today.',
                    'corpus-complete':'All available unique examples assigned.',
                    'review-pending':'Submitted models are awaiting review.',
                }
                print(details.get(reason,'Training queue temporarily unavailable.'),flush=True)
                break
            job=r.json();assignment=job.get('corpus')
            if args.teacher and not args.no_teacher and not teacher_attempted:
                teacher_attempted=True
                try:
                    wait(800)
                    from trainer.teacher import Teacher
                    print('Loading cached pretrained checker for teacher-assisted lessons.',flush=True)
                    teacher=Teacher('cpu')
                except Exception as error:
                    print('Pretrained checker unavailable ('+type(error).__name__+
                          '). Training with original dataset lessons instead.',flush=True)
                    teacher=None
            if not assignment or assignment['id']!=info['id'] or assignment['revision']!=info['revision']:raise RuntimeError('Server does not support this corpus client yet.')
            arch=job['manifest']['architecture']
            if arch!=model.config:raise RuntimeError('Student architecture mismatch')
            rows=[]
            for offset,raw in enumerate(read_assigned(args.cache,pages,assignment['start'],assignment['count'])):
                print(f"Preparing record {assignment['start']+offset+1}…",flush=True)
                wait(256)
                from trainer.corpus import training_context
                context, answer = training_context(raw)
                if teacher is not None and not teacher.review_source(raw):
                    print('  Original record failed two review checks; skipped.',flush=True)
                    continue
                budget = 509 - sum(len(m['content'].encode()) + 2 for m in context)
                if context and answer and budget > 24:
                    if len(answer.encode()) > budget:
                        answer = answer.encode()[:budget].decode('utf-8', errors='ignore').rsplit(' ', 1)[0].strip()
                    if answer:
                        rows.append({'concept':'conversation', 'messages':context, 'response':answer})
                if teacher:
                    # The global corpus reservation gives every PC different seeds.
                    # Only twice-approved generated records enter student training.
                    item=teacher.generate_checked(raw,assignment['start']+offset)
                    if item:
                        rows.append(item)
                        with (Path(args.cache)/'verified-generated.jsonl').open('a',encoding='utf-8') as output:
                            output.write(json.dumps({'reservation':assignment['start']+offset,
                                'checks':2,'row':item},ensure_ascii=False)+'\n')
                    else:
                        print('  Generated candidate rejected by local review.',flush=True)
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
                train_model(model,rows,args.device,before_step=lambda:wait(256))
                save_file(snapshot(model),str(file));data=file.read_bytes();chunk_size=2_500_000
                parts=(len(data)+chunk_size-1)//chunk_size
                for index in range(parts):
                    request(session,'POST',base+'/api/update-part',data=data[index*chunk_size:(index+1)*chunk_size],headers={'Content-Type':'application/octet-stream','X-AXON-Job':job['id'],'X-AXON-Token':job['upload_token'],'X-AXON-Part':str(index)})
                result=request(session,'POST',base+'/api/update',json={'job':job['id'],'token':job['upload_token'],'parts':parts,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                print(f'Round {n+1}: {len(rows)} lessons trained using full backpropagation; checkpoint submitted.',flush=True)
    finally:
        session.close()
        if teacher is not None: teacher.close()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nStopped.')
    except MemoryAdmissionError as error:
        print('\n'+str(error),flush=True)
        sys.exit(2)
