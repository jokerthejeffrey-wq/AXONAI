import sqlite3,json,hashlib,re,contextlib,time
from settings import STATE,ROOT,MAX_USES,manifest

@contextlib.contextmanager
def connect():
    STATE.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(STATE/'student.db',timeout=30);c.row_factory=sqlite3.Row
    c.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS concepts(name TEXT PRIMARY KEY,uses INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS lessons(id TEXT PRIMARY KEY,concept TEXT,payload TEXT,uses INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,version INTEGER,created REAL,used INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS assigned(job TEXT,lesson TEXT);
    CREATE TABLE IF NOT EXISTS updates(id TEXT PRIMARY KEY,version INTEGER,status TEXT,created REAL);''')
    c.execute("INSERT OR IGNORE INTO meta VALUES('version','0')");c.commit()
    try:
        yield c;c.commit()
    except BaseException:c.rollback();raise
    finally:c.close()

def version():
    with connect() as c:return int(c.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0])
def checkpoint(v):return STATE/'checkpoints'/f'{v}.safetensors'
def normalized(s):return ' '.join(re.sub(r'[^\w\s]',' ',s.casefold()).split())
def concept_name(row):
    concept=normalized(row['concept']).replace(' ','_')
    prompt=normalized(row['messages'][-1]['content'])
    # Known greeting aliases are combined even if an author mistags them.
    if concept in ('hi','hello','greeting','greetings','salutation','salutations') or re.fullmatch(r'(hi|hey|hello|greetings|good morning|good evening|good afternoon)( there| everyone| axonai)?',prompt):return 'greetings'
    return concept

def import_lessons():
    rows=[json.loads(s) for s in (ROOT/'data/train.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    with connect() as c:
        signature=json.dumps(manifest(),sort_keys=True)
        old=c.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if old and old[0]!=signature:raise ValueError('Configuration changed: use a NEW AXON_STATE')
        c.execute("INSERT OR IGNORE INTO meta VALUES('manifest',?)",(signature,))
        for row in rows:
            if not row.get('response') or not row.get('messages'):raise ValueError('Invalid training row')
            concept=concept_name(row);row['concept']=concept
            # Context fingerprint: editing the answer cannot reset the prompt's use counter.
            key=json.dumps([(m['role'],normalized(m['content'])) for m in row['messages']],ensure_ascii=False)
            identity=hashlib.sha256(key.encode()).hexdigest()
            c.execute('INSERT OR IGNORE INTO concepts(name) VALUES(?)',(concept,))
            c.execute('INSERT OR IGNORE INTO lessons(id,concept,payload) VALUES(?,?,?)',(identity,concept,json.dumps(row)))
    return len(rows)
