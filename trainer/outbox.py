"""Durable checkpoint outbox: preserve completed volunteer work across upload outages.

Data is written atomically before the first network upload. Replaying the same
authorized job is idempotent at the server; never retrain or claim a new range
until the pending checkpoint has been acknowledged.
"""
import hashlib,json,os,re,time
from pathlib import Path
from urllib.parse import urlparse
import requests
from trainer.progress import bar

PART_BYTES=2_500_000
_JOB=re.compile(r'^[a-f0-9]{32}$')
_TOKEN=re.compile(r'^[a-f0-9]{64}$')
_RETRY={429,500,502,503,504,507}

def _secure_write(path,data):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'wb') as file:
        try:os.chmod(tmp,0o600)
        except OSError:pass
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp,path)

def save_checkpoint(cache,job,server,data):
    if not isinstance(data,bytes) or not 1000<len(data)<=32*1024*1024:
        raise ValueError('Invalid checkpoint data')
    ident=job.get('id')
    token=job.get('upload_token')
    if not _JOB.fullmatch(str(ident)) or not _TOKEN.fullmatch(str(token)):
        raise ValueError('Invalid checkpoint job authorization')
    root=Path(cache)/'outbox'
    root.mkdir(parents=True,exist_ok=True)
    state={'job':ident,'token':token,'server':server.rstrip('/'),
           'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),
           'parts':(len(data)+PART_BYTES-1)//PART_BYTES}
    checkpoint=root/(ident+'.safetensors')
    manifest=root/(ident+'.json')
    _secure_write(checkpoint,data)
    _secure_write(manifest,json.dumps(state,separators=(',',':')).encode('utf-8'))
    print('Completed checkpoint saved locally before upload: '+str(checkpoint),flush=True)
    return manifest

def _remote_state(session,server,state):
    reply=session.request('POST',server+'/api/job-state',
                          json={'job':state['job'],'token':state['token']},
                          timeout=(10,60))
    if reply.status_code!=200:
        return None
    remote=reply.json()
    if (remote.get('status') in ('pending','approved') and
        remote.get('sha256')==state['sha256'] and
        remote.get('size')==state['size']):
        return 'received'
    if remote.get('status')=='rejected':
        return 'rejected'
    return None

def _request(session,method,url,**kwargs):
    return session.request(method,url,timeout=(15,120),**kwargs)

def _send(session,server,state,payload,log=print):
    chunk_count=state['parts']
    headers={'Content-Type':'application/octet-stream',
             'X-AXON-Job':state['job'],'X-AXON-Token':state['token']}
    for index in range(chunk_count):
        chunk=payload[index*PART_BYTES:(index+1)*PART_BYTES]
        reply=_request(session,'POST',server+'/api/update-part',
                       data=chunk,headers={**headers,'X-AXON-Part':str(index)})
        if reply.status_code==409 and _remote_state(session,server,state)=='received':
            return 'received'
        if reply.status_code!=200:
            return reply
        log(bar(round(100*(index+1)/chunk_count),'Uploading checkpoint'),flush=True)
    reply=_request(session,'POST',server+'/api/update',
                   json={'job':state['job'],'token':state['token'],
                         'parts':chunk_count,'size':state['size'],'sha256':state['sha256']})
    if reply.status_code==409 and _remote_state(session,server,state)=='received':
        return 'received'
    return 'received' if reply.status_code==200 else reply

def flush_pending(cache,session,server,sleep=time.sleep,log=print):
    root=Path(cache)/'outbox'
    if not root.exists():
        return 0
    delivered=0
    for manifest in sorted(root.glob('*.json')):
        state=json.loads(manifest.read_text(encoding='utf-8'))
        if state['server']!=server.rstrip('/'):
            raise RuntimeError('Pending checkpoint belongs to a different coordinator: '+str(manifest))
        ident=state['job']
        if not _JOB.fullmatch(ident) or not _TOKEN.fullmatch(state.get('token','')):
            raise RuntimeError('Invalid saved checkpoint authorization; not uploading.')
        path=root/(ident+'.safetensors')
        data=path.read_bytes()
        if len(data)!=state['size'] or hashlib.sha256(data).hexdigest()!=state['sha256']:
            raise RuntimeError('Local pending checkpoint failed SHA-256 verification: '+str(path))
        log('Resuming checkpoint upload for job '+ident[:8]+' (data already trained).',flush=True)
        while True:
            try:
                answer=_send(session,server,state,data,log)
                if answer=='received':
                    manifest.unlink()
                    path.unlink()
                    delivered+=1
                    log('Checkpoint accepted into the review queue. Locally saved copy cleared.',flush=True)
                    break
                code=answer.status_code
                if code==409:
                    remote=_remote_state(session,server,state)
                    if remote=='rejected':
                        raise RuntimeError('Checkpoint rejected by coordinator; saved file kept: '+str(path))
                if code not in _RETRY:
                    raise RuntimeError('Checkpoint upload HTTP '+str(code)+'; saved file kept: '+str(path))
                try:
                    body=answer.json()
                    detail=str(body.get('error',''))
                except (ValueError,AttributeError):
                    detail=''
                delay=150 if code==507 else 45
                log(('Vercel Blob storage full; ' if code==507 else 'Upload service temporarily unavailable; ')+
                    'trained checkpoint is SAFE on disk. Retrying in '+str(delay)+' seconds. '+detail[:140],flush=True)
            except (requests.Timeout,requests.ConnectionError):
                delay=45
                log('Network interrupted; trained checkpoint saved. Retrying upload in 45 seconds.',flush=True)
            sleep(delay)
    return delivered
