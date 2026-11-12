"""Persistent Flask backend; static website can also be deployed to Vercel."""
if __name__=='__main__':
    from bootstrap import ensure
    ensure('teacher','HTML.py')
import io,os,secrets,time,threading,zipfile,json
from collections import defaultdict,deque
from functools import wraps
from flask import Flask,request,jsonify,abort,render_template,send_file
from settings import ROOT,STATE,manifest,MAX_USES,BATCH_LESSONS
from ledger import connect,version,checkpoint
from CHAT import ChatEngine
app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=32*1024*1024
engine=ChatEngine();chat_slot=threading.BoundedSemaphore(1);rates=defaultdict(deque);rate_lock=threading.Lock()

def limited(name,count,seconds):
    def deco(fn):
        @wraps(fn)
        def wrapped(*a,**kw):
            now=time.monotonic();key=(name,request.remote_addr)
            with rate_lock:
                if len(rates)>10000:
                    for k in list(rates):
                        if not rates[k] or now-rates[k][-1]>3600:del rates[k]
                    if len(rates)>10000:abort(429)
                q=rates[key]
                while q and q[0]<now-seconds:q.popleft()
                if len(q)>=count:abort(429,'Too many requests; retry later.')
                q.append(now)
            return fn(*a,**kw)
        return wrapped
    return deco

def authorized():
    token=os.getenv('AXON_TRAIN_TOKEN','')
    if token and not secrets.compare_digest(request.headers.get('Authorization',''),'Bearer '+token):abort(401,'Contributor token required.')

@app.before_request
def cors_check():
    origin=request.headers.get('Origin')
    allowed={s.strip().rstrip('/') for s in os.getenv('AXON_ALLOWED_ORIGINS','').split(',') if s.strip()}
    if origin and origin not in {request.host_url.rstrip('/'),'https://'+request.host} and origin not in allowed:abort(403,'Frontend origin not allowed by backend.')
    if request.method=='OPTIONS':return '',204

@app.after_request
def headers(r):
    origin=request.headers.get('Origin')
    allowed={s.strip().rstrip('/') for s in os.getenv('AXON_ALLOWED_ORIGINS','').split(',') if s.strip()}
    if origin and (origin in allowed or origin in {request.host_url.rstrip('/'),'https://'+request.host}):
        r.headers['Access-Control-Allow-Origin']=origin;r.headers['Vary']='Origin'
        r.headers['Access-Control-Allow-Headers']='Content-Type';r.headers['Access-Control-Allow-Methods']='GET,POST,OPTIONS'
    r.headers['X-Content-Type-Options']='nosniff';r.headers['X-Frame-Options']='DENY'
    if request.path.startswith('/api/'):r.headers['Cache-Control']='no-store'
    return r

for code in [400,401,403,404,409,413,429,503]:
    app.register_error_handler(code,lambda e:(jsonify(error=e.description),e.code))

@app.get('/')
def home():return render_template('index.html')
@app.get('/chat')
def chat_page():return render_template('chat.html')
@app.get('/docs')
def docs():return render_template('docs.html')
@app.get('/config.js')
def config():return app.response_class('window.AXON_API="";',mimetype='application/javascript')
@app.get('/health')
def health():return {'ok':True}
@app.get('/api/status')
def status():
    v=version()
    with connect() as c:
        counts=dict(c.execute('SELECT status,count(*) FROM updates GROUP BY status').fetchall())
        remaining=c.execute('SELECT coalesce(sum(max(0,?-uses)),0) FROM concepts',(MAX_USES,)).fetchone()[0]
    return dict(version=v,pending=counts.get('pending',0),accepted=counts.get('accepted',0),remaining=remaining,chat_mode=engine.mode,ready=checkpoint(v).exists(),model='AXONAI byte transformer · from scratch')

@app.post('/api/chat')
@limited('chat',12,60)
def chat():
    body=request.get_json(silent=True);msgs=body.get('messages') if isinstance(body,dict) else None
    if not isinstance(msgs,list) or not 1<=len(msgs)<=13:abort(400,'Send 1–13 messages.')
    for i,m in enumerate(msgs):
        if not isinstance(m,dict) or m.get('role')!=('user' if i%2==0 else 'assistant'):abort(400,'Messages must alternate user and assistant.')
        if not isinstance(m.get('content'),str) or not 1<=len(m['content'])<=4000:abort(400,'Message length must be 1–4000 characters.')
    if msgs[-1]['role']!='user':abort(400,'Last message must be from the user.')
    if not chat_slot.acquire(False):abort(429,'The student is answering another request. Retry shortly.')
    try:return {'reply':engine.chat(msgs),'version':engine.loaded_version,'mode':engine.mode}
    except RuntimeError as e:abort(503,str(e))
    except Exception:
        app.logger.exception('Student inference failed');abort(503,'Student unavailable. Owner: inspect the server console.')
    finally:chat_slot.release()

@app.post('/api/job')
@limited('jobs',12,3600)
def job():
    authorized()
    if not checkpoint(version()).exists():abort(503,'Owner must generate lessons and run manage.py init.')
    jid=secrets.token_hex(16);now=time.time();rows=[]
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        v=int(c.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0])
        if c.execute('SELECT count(*) FROM jobs WHERE created>?',(now-172800,)).fetchone()[0]>=1000:abort(429,'Job capacity reached.')
        c.execute('INSERT INTO jobs(id,version,created) VALUES(?,?,?)',(jid,v,now))
        for _ in range(BATCH_LESSONS):
            row=c.execute('''SELECT l.* FROM lessons l JOIN concepts c ON l.concept=c.name
            WHERE l.uses<? AND c.uses<? ORDER BY l.uses,c.uses,l.rowid LIMIT 1''',(MAX_USES,MAX_USES)).fetchone()
            if not row:break
            c.execute('UPDATE lessons SET uses=uses+1 WHERE id=?',(row['id'],))
            c.execute('UPDATE concepts SET uses=uses+1 WHERE name=?',(row['concept'],))
            c.execute('INSERT INTO assigned VALUES(?,?)',(jid,row['id']))
            rows.append(json.loads(row['payload']))
        if not rows:
            c.execute('DELETE FROM jobs WHERE id=?',(jid,));return '',204
    # A reserved use is NEVER refunded, even on disconnect, rejection, expiry or rollback.
    return {'id':jid,'version':v,'rows':rows,'manifest':manifest()}

@app.get('/api/checkpoint/<int:v>')
def download_checkpoint(v):
    if not checkpoint(v).exists():abort(404)
    return send_file(checkpoint(v),as_attachment=True,download_name=f'axon-student-{v}.safetensors')

@app.post('/api/update')
@limited('uploads',12,3600)
def update():
    authorized();jid=request.form.get('job','');f=request.files.get('weights')
    if not f or len(jid)!=32:abort(400,'Missing job or weights.')
    raw=f.read(32*1024*1024+1)
    if len(raw)>32*1024*1024:abort(413)
    import numpy as np
    from safetensors.numpy import load,load_file
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        j=c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
        v=int(c.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0])
        if not j or j['used'] or j['version']!=v or time.time()-j['created']>172800:abort(409,'Expired, stale, unknown, or used job.')
        if c.execute("SELECT count(*) FROM updates WHERE status='pending'").fetchone()[0]>=100:abort(429,'Review queue full.')
        try:
            incoming=load(raw);expected=load_file(str(checkpoint(v)));squares=0.;count=0
            if incoming.keys()!=expected.keys():raise ValueError()
            for key,t in incoming.items():
                if t.dtype!=np.float32 or t.shape!=expected[key].shape or not np.isfinite(t).all():raise ValueError()
                squares+=float(np.square(t.astype(np.float64)-expected[key]).sum());count+=t.size
            if not np.isfinite(squares) or squares/max(1,count)>.05**2:raise ValueError()
        except Exception:abort(400,'Invalid tensors or excessive update magnitude.')
        uid=secrets.token_hex(16);folder=STATE/'pending';folder.mkdir(parents=True,exist_ok=True)
        (folder/f'{uid}.safetensors').write_bytes(raw)
        c.execute('INSERT INTO updates VALUES(?,?,?,?)',(uid,v,'pending',time.time()))
        c.execute('UPDATE jobs SET used=1 WHERE id=?',(jid,))
    return {'id':uid,'status':'pending'},201

@app.get('/download/source/<name>')
@limited('repair',60,60)
def repair_file(name):
    allowed={'GPU.py','CPU.py','bootstrap.py','start.ps1','START_CPU.cmd','START_GPU.cmd','client.py','student.py','settings.py','requirements.txt','CLIENT_README.md'}
    if name not in allowed:abort(404)
    return send_file(ROOT/name,mimetype='application/octet-stream',as_attachment=True)

@app.get('/download/client.zip')
@limited('downloads',30,60)
def client_zip():
    names=['client_manifest.json','GPU.py','CPU.py','bootstrap.py','start.ps1','START_CPU.cmd','START_GPU.cmd','client.py','student.py','settings.py','requirements.txt','CLIENT_README.md']
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,'AXONAI_Client/'+name)
    out.seek(0);return send_file(out,mimetype='application/zip',as_attachment=True,download_name='AXONAI_Client.zip')

if __name__=='__main__':
    from waitress import serve
    print('AXONAI backend: http://127.0.0.1:5000')
    serve(app,host=os.getenv('HOST','127.0.0.1'),port=int(os.getenv('PORT','5000')),threads=4)
