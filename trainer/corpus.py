"""Disk-backed, deterministic conversation corpus. Never pad with duplicate lines."""
import hashlib,json,sqlite3,os,gc
from pathlib import Path
DATASET='HuggingFaceTB/smol-smoltalk'
REVISION='2ca82d2fe63c017af8d4cb30e81e95cb25516851'
CORPUS_ID='smoltalk-v1'
TARGET_BYTES=1_000_000_000

def training_context(row):
    history=[];target=''
    for m in row.get('messages',[]):
        if m.get('role')=='user':
            history.append({'role':'user','content':m['content']});target=''
        elif m.get('role')=='assistant' and history:
            target=m['content'];history.append({'role':'assistant','content':target})
    if not history:return [],''
    if history[-1]['role']=='assistant':history.pop()
    while history and (history[0]['role']!='user' or sum(len(m['content'].encode())+2 for m in history)>220):
        history.pop(0)
    if not history or any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(history)) or history[-1]['role']!='user':return [],''
    return history,target

def canonical(row):
    messages=row.get('messages',[])
    if not isinstance(messages,list) or len(messages)<2:return None
    if any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) for m in messages):return None
    if not training_context(row)[0]:return None
    return {'messages':messages}

def _prepare(folder,source=None,target_bytes=TARGET_BYTES):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'corpus.json';txt=folder/'training.txt';dbpath=folder/'index.sqlite'
    if meta.exists() and txt.exists() and dbpath.exists():
        info=json.loads(meta.read_text())
        if info.get('revision')==REVISION and info.get('target_bytes')==target_bytes and txt.stat().st_size==info['bytes']:
            digest=hashlib.sha256()
            with txt.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=info['sha256']:raise RuntimeError('Corpus checksum failed; use a new cache directory.')
            return info
        raise RuntimeError('Corpus cache mismatch. Use a new cache directory; existing files were preserved.')
    if source is None:
        from datasets import load_dataset
        source=load_dataset(DATASET,split='train',revision=REVISION,streaming=True)
    # A partial build is safely reconstructed. Completion is indicated only by corpus.json.
    db=sqlite3.connect(dbpath)
    db.execute('DROP TABLE IF EXISTS records')
    db.execute('CREATE TABLE records(id INTEGER PRIMARY KEY, offset INTEGER, size INTEGER, fingerprint TEXT UNIQUE)')
    count=size=0;digest=hashlib.sha256();iterator=iter(source)
    try:
        with txt.open('wb') as f:
            for raw in iterator:
                row=canonical(raw)
                if row is None:continue
                # Deduplicate whole user contexts even if the source changes the answer.
                prompts=[(m['role'],' '.join(m['content'].casefold().split())) for m in training_context(row)[0]]
                key=hashlib.sha256(json.dumps(prompts,ensure_ascii=False).encode()).hexdigest()
                if db.execute('SELECT 1 FROM records WHERE fingerprint=?',(key,)).fetchone():continue
                data=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                if size+len(data)>target_bytes:break
                db.execute('INSERT INTO records VALUES(?,?,?,?)',(count,size,len(data),key));f.write(data);digest.update(data)
                size+=len(data);count+=1
                if count%1000==0:db.commit();print(f'Corpus: {count:,} conversations, {size/1e6:.1f} MB',flush=True)
        db.commit()
    finally:
        db.close()
        close=getattr(iterator,'close',None)
        if close:close()
        del iterator,source
        gc.collect()
    info={'id':CORPUS_ID,'dataset':DATASET,'revision':REVISION,'target_bytes':target_bytes,'bytes':size,'records':count,'lines':count,'sha256':digest.hexdigest()}
    temp=meta.with_suffix('.tmp');temp.write_text(json.dumps(info,indent=2));temp.replace(meta)
    print(f"Corpus ready: {count:,} distinct conversations, {size:,} bytes. This is NOT 15 million examples.",flush=True)
    return info

def read_rows(folder,start,count):
    folder=Path(folder)
    with sqlite3.connect(folder/'index.sqlite') as db, (folder/'training.txt').open('rb') as f:
        rows=[]
        for i in range(start,start+count):
            entry=db.execute('SELECT offset,size FROM records WHERE id=?',(i,)).fetchone()
            if entry is None:raise RuntimeError('Corpus range exhausted or cache incomplete')
            f.seek(entry[0]);rows.append(json.loads(f.read(entry[1])))
        return rows


def prepare(folder,source=None,target_bytes=TARGET_BYTES):
    if source is not None:return _prepare(folder,source,target_bytes)
    from filelock import FileLock
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # CPU and GPU launches on the same PC must not write the same cache concurrently.
    with FileLock(str(folder/'corpus.lock'),timeout=3600):
        return _prepare(folder,None,target_bytes)
