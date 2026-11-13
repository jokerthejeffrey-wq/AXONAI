"""Disk-backed, deterministic conversation corpus. Never pad with duplicate lines."""
import hashlib,json,sqlite3,os,gc,time
from pathlib import Path
DATASET='HuggingFaceTB/smol-smoltalk'
REVISION='2ca82d2fe63c017af8d4cb30e81e95cb25516851'
CORPUS_ID='smoltalk-v1'
TARGET_BYTES=1_000_000_000

class Progress:
    """Single-line progress based on bytes processed since the start."""
    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.start = time.monotonic()
        self.last_update = 0
        self.width = 0
        self.shown = False

    def update(self, current, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < 1:
            return
        self.last_update = now
        percent = min(100, max(0, int(100 * current / self.total)))
        filled = percent // 5
        elapsed = now - self.start
        if current > 0 and elapsed >= 2:
            seconds = max(0, int((self.total - current) * elapsed / current))
            eta = f'~{seconds // 60}m {seconds % 60}s' if seconds >= 60 else f'~{seconds}s'
        else:
            eta = 'estimating'
        line = f'{self.label} [{"=" * filled}{" " * (20 - filled)}] {percent}% ETA {eta}'
        print('\r' + line.ljust(self.width), end='', flush=True)
        self.width = len(line)
        self.shown = True

    def finish(self):
        if self.shown:
            print(flush=True)
            self.shown = False


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

# Pinned train shard hashes preserve the original dataset record order.
SHARDS = (
    '498cd4580014f42c40cfde066573564ed28e3e1548a5566715b615f5ea932856',
    '5412c03df579fc7dd7d06c7f5249627d99d0621882fb8f2cd5c13582eb2a80f6',
    'aefcac6571abf275b4e1629baa01f326c0fe279d35cd0349b7439319923815a7',
    'c15edfca888c9924a626dad7b7f4e9f08d8f406f997ac13be46bc9be0a6192db',
)

def dataset_rows(folder, corpus_progress=None):
    import requests
    import pyarrow.parquet as parquet
    import tempfile
    for index, expected in enumerate(SHARDS):
        name = f'train-{index:05d}-of-00004.parquet'
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}'
        if corpus_progress:
            corpus_progress.finish()
        print(f'Downloading training data shard {index + 1}/4…', flush=True)
        with tempfile.TemporaryDirectory(prefix='axon-data-', dir=folder) as temp:
            path = Path(temp) / name
            digest = hashlib.sha256()
            downloaded = 0
            download_progress = Progress('Download', (229710350, 230298184, 230587255, 231879938)[index])
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with path.open('wb') as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
                            digest.update(block)
                            downloaded += len(block)
                            download_progress.update(downloaded)
            download_progress.update(downloaded, force=True)
            download_progress.finish()
            if digest.hexdigest() != expected:
                raise RuntimeError(f'Data checksum failed for {name}. Retry the download.')
            reader = parquet.ParquetFile(path)
            try:
                for batch in reader.iter_batches(batch_size=256, columns=['messages']):
                    yield from batch.to_pylist()
            finally:
                reader.close()


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
    progress=Progress('Corpus', target_bytes)
    if source is None:
        source=dataset_rows(folder, progress)
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
                if count%1000==0:
                    db.commit()
                    progress.update(size)
        db.commit()
    finally:
        progress.finish()
        db.close()
        close=getattr(iterator,'close',None)
        if close:close()
        del iterator,source
        gc.collect()
    info={'id':CORPUS_ID,'dataset':DATASET,'revision':REVISION,'target_bytes':target_bytes,'bytes':size,'records':count,'lines':count,'sha256':digest.hexdigest()}
    temp=meta.with_suffix('.tmp');temp.write_text(json.dumps(info,indent=2));temp.replace(meta)
    print(f"Corpus ready: {count:,} conversations, {size:,} bytes.",flush=True)
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
