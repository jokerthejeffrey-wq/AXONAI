"""Full-gradient transformer trainer with on-demand verified lesson pages."""
import argparse,os,sys,json,hashlib,time,uuid,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from trainer.chunked_corpus import load_manifest, read_assigned
from trainer.memory import wait, MemoryAdmissionError
from trainer.assignments import rounds, claim_until_ready
from trainer.lessons import source_lessons
from trainer.progress import bar
from trainer.outbox import save_checkpoint,flush_pending

def request(session,method,url,**kwargs):
    for attempt in range(10):
        r=session.request(method,url,timeout=300,**kwargs)
        if r.status_code not in (429,503):r.raise_for_status();return r
        print('Server busy; retrying in 30 seconds.',flush=True);time.sleep(30)
    raise RuntimeError('Server remains busy. Restart later; completed reservations will not be reassigned.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--server',default='https://axonai-theta.vercel.app');p.add_argument('--rounds',type=int,default=0,help='0 = unlimited session, the default')
    p.add_argument('--teacher', action='store_true', help='Use built-in source-grounded local lesson helper (default)')
    p.add_argument('--cache',default=str(Path.home()/'.axonai'/'smoltalk-v1'));p.add_argument('--yes',action='store_true');p.add_argument('--no-teacher',action='store_true');args=p.parse_args()
    from urllib.parse import urlparse
    parsed=urlparse(args.server)
    if parsed.scheme!='https' or parsed.username or parsed.password:p.error('An HTTPS coordinator URL is required')
    if args.rounds<0:p.error('Rounds must be 0 (unlimited) or positive')
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
    print('Local teacher is non-pretrained and requires zero model downloads.', flush=True)
    print('No daily cap: runs continuously until Ctrl+C or unique data runs out.', flush=True)
    print('A busy review queue pauses the trainer instead of ending the session.', flush=True)
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
    trained_total=0
    try:
        # Resume previously trained weights BEFORE reserving another unique job.
        flush_pending(args.cache,session,base)
        model=Student().to(args.device)
        for n in rounds(args.rounds):
            wait(512)
            r=claim_until_ready(lambda:request(session,'POST',base+'/api/job',
                                                  json={'worker':worker,'corpus':info}))
            if r is None:break
            job=r.json();assignment=job.get('corpus')
            if not args.no_teacher and not teacher_attempted:
                teacher_attempted=True
                try:
                    wait(800)
                    from trainer.teacher import Teacher
                    print('Starting built-in local source-grounded lesson helper.',flush=True)
                    teacher=Teacher('cpu')
                except Exception as error:
                    print('Local lesson helper unavailable ('+type(error).__name__+
                          '). Training with original dataset lessons instead.',flush=True)
                    teacher=None
            if not assignment or assignment['id']!=info['id'] or assignment['revision']!=info['revision']:raise RuntimeError('Server does not support this corpus client yet.')
            arch=job['manifest']['architecture']
            if arch!=model.config:raise RuntimeError('Student architecture mismatch')
            rows=[]
            used_records=0
            source_skips=0
            for offset,raw in enumerate(read_assigned(args.cache,pages,assignment['start'],assignment['count'])):
                wait(256)
                prepared=source_lessons(raw)
                if prepared:
                    used_records+=1
                    rows.extend(prepared)
                else:
                    source_skips+=1
                if teacher and teacher.review_source(raw):
                    # Optional local variation is strictly in addition to
                    # original training examples; it cannot veto them.
                    item=teacher.generate_checked(raw,assignment['start']+offset)
                    if item and len(rows)<assignment['count']*4:
                        rows.append(item)
                        with (Path(args.cache)/'verified-generated.jsonl').open('a',encoding='utf-8') as output:
                            output.write(json.dumps({'reservation':assignment['start']+offset,
                                'checks':'local-source-grounded','row':item},ensure_ascii=False)+'\n')
                print(bar(round(100*(offset+1)/assignment['count']),
                          'Preparing round '+str(n+1)),flush=True)
            print(f'Round {n+1}: {used_records}/{assignment["count"]} records usable, '
                  f'{len(rows)} lessons prepared, {source_skips} skipped.',flush=True)
            if not rows:
                print('No safe short examples in this reservation; nothing submitted.',flush=True)
                continue
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
                save_file(snapshot(model),str(file))
                data=file.read_bytes()
                save_checkpoint(args.cache,job,base,data)
                flush_pending(args.cache,session,base)
                trained_total+=len(rows)
                print(f'Round {n+1} complete: {len(rows)} local optimizer steps, '
                      f'{trained_total} total this session. Checkpoint SUBMITTED for '
                      f'server review, not yet approved.',flush=True)
                if n % 5==0:
                    try:
                        progress=request(session,'GET',base+'/api/status').json()
                        print(f'SHARED MODEL: version {progress.get("training_version", "?")}'
                              f' approved; {progress.get("pending_updates", "?")}'
                              f' pending. Local lessons trained: {trained_total}.',flush=True)
                    except Exception:
                        print('Shared version temporarily unavailable; local training continues.',flush=True)
    finally:
        session.close()
        if teacher is not None: teacher.close()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nStopped.')
    except MemoryAdmissionError as error:
        print('\n'+str(error),flush=True)
        sys.exit(2)
