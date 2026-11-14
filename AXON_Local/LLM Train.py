"""AXON human/assistant parallel conversation training.

Several simulated humans talk with a locally hosted Qwen assistant in parallel.
Human and Assistant are two independent local model processes (same GGUF weights).
AXON notes human messages and assistant answers, and trains only on correct
human -> assistant pairs. Its normal chat generates its own tokens using NumPy.

This is an experimental 25M-parameter next-token model, NOT a transformer.
It may need substantially more and better training to write coherent replies.
Four project files; downloadable runtime cache is outside the project folder.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import concurrent.futures
import hashlib
import importlib
import json
import math
import uuid
import os
from pathlib import Path
import platform
import queue
import random
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'BRAIN.db'
DATA_PATH = ROOT / 'training_data.txt'
CACHE = Path.home() / '.axon_four_files_cache'
GGUF = CACHE / 'qwen2.5-0.5b-instruct-q4_k_m.gguf'
TEACHER_URL = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'
TEACHER_SHA = '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
MAX_DB_BYTES = 15 * 1024**3
MAX_MODEL_BYTES = 2 * 1024**3
VOCAB_SIZE = 16384
DIM = 768
PARAMETERS = 2 * VOCAB_SIZE * DIM
USER_AGENT = 'AXON-FourFile/1.1 (local educational LLM)'
TOKEN_RE = re.compile(r"<user>|<assistant>|<eos>|[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)


def say(message=''):
    print(message, flush=True)


def progress(name, done, total, began):
    elapsed = max(time.monotonic() - began, .001)
    rate = done / elapsed
    if total:
        percent = min(100.0, done / total * 100)
        bars = int(percent / 5)
        remaining = max(0, (total - done) / max(rate, .00001))
        eta = f'{int(remaining // 60)}m {int(remaining % 60):02d}s'
        print(f'\r{name} [{"=" * bars}{"." * (20 - bars)}] {percent:5.1f}%  ETA {eta} ', end='', flush=True)
    else:
        print(f'\r{name} {done / 1048576:.1f} MiB  {rate / 1048576:.1f} MiB/s ', end='', flush=True)


def connection():
    db = sqlite3.connect(str(DB_PATH), timeout=120)
    db.execute('PRAGMA busy_timeout=120000')
    db.execute('PRAGMA journal_mode=DELETE')  # no permanent WAL/SHM project files
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, val TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS weights (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS models (name TEXT PRIMARY KEY, sha256 TEXT, bytes INTEGER NOT NULL, data BLOB NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS lessons (id INTEGER PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL, origin TEXT NOT NULL, created INTEGER NOT NULL)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS lessons_unique ON lessons(prompt,answer)')
    db.execute('CREATE INDEX IF NOT EXISTS lessons_prompt ON lessons(prompt)')
    db.execute('CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, session TEXT NOT NULL, turn INTEGER NOT NULL, speaker TEXT NOT NULL, receiver TEXT NOT NULL, replied_to TEXT NOT NULL, message TEXT NOT NULL, created INTEGER NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS obs_session ON observations(session,turn)')
    db.execute('CREATE TABLE IF NOT EXISTS nextword (context TEXT NOT NULL, token TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(context,token))')
    db.commit()
    return db


def set_meta(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta(key,val) VALUES (?,?)', (key, str(value)))


def get_meta(db, key, default=None):
    row = db.execute('SELECT val FROM meta WHERE key=?', (key,)).fetchone()
    return row[0] if row else default


def enforce_db_size(db, incoming=0):
    pages = db.execute('PRAGMA page_count').fetchone()[0]
    size = db.execute('PRAGMA page_size').fetchone()[0]
    if pages * size + incoming > MAX_DB_BYTES:
        raise RuntimeError('BRAIN.db would exceed the strict 15 GiB limit; training stopped safely.')


def store_lesson(db, question, answer, origin='corpus'):
    question, answer = question.strip()[:1500], answer.strip()[:2500]
    if not question or not answer:
        return
    db.execute('INSERT OR IGNORE INTO lessons(prompt,answer,origin,created) VALUES(?,?,?,?)', (question, answer, origin, int(time.time())))


def tokenizer(text):
    return TOKEN_RE.findall(text.lower())


def ensure_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        say('Installing NumPy (no PyTorch needed)...')
        proc = subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', 'numpy>=1.26'], check=False)
        if proc.returncode:
            raise RuntimeError('NumPy install failed. Choose Python 3.11–3.13 (64-bit) in PyCharm and retry.')
        return importlib.import_module('numpy')


def request(url):
    return urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/octet-stream'})


def download(url, target, limit, label='Download'):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.partial')
    # Don't resume against URLs whose content/ETag could have changed.
    if part.exists():
        part.unlink()
    began = time.monotonic()
    done = 0
    try:
        with urllib.request.urlopen(request(url), timeout=65) as src:
            size = int(src.headers.get('Content-Length', 0) or 0)
            if size > limit:
                raise RuntimeError(f'{label} too large: {size} > {limit} bytes')
            with part.open('wb') as out:
                last = time.monotonic()
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    done += len(block)
                    if done > limit:
                        raise RuntimeError(f'{label} exceeded byte limit')
                    out.write(block)
                    if time.monotonic() - last >= .3:
                        progress(label, done, size, began)
                        last = time.monotonic()
        if size and done != size:
            raise RuntimeError(f'Incomplete download: got {done} expected {size}')
        part.replace(target)
        progress(label, done, size, began)
        say()
        return done
    except Exception:
        part.unlink(missing_ok=True)
        raise


def cached_valid(db):
    row = db.execute('SELECT bytes FROM models WHERE name=?', ('teacher',)).fetchone()
    return bool(row and GGUF.is_file() and GGUF.stat().st_size == row[0] and get_meta(db, 'teacher_sha') == TEACHER_SHA)


def write_model_to_db(db, filename):
    size = Path(filename).stat().st_size
    if size > MAX_MODEL_BYTES:
        raise RuntimeError('Pretrained model larger than expected')
    enforce_db_size(db, size + 16*1024*1024)
    say('Storing teacher weights inside BRAIN.db...')
    db.execute('DELETE FROM models WHERE name=?', ('teacher',))
    db.execute('INSERT INTO models(name,sha256,bytes,data) VALUES(?,?,?,zeroblob(?))', ('teacher',TEACHER_SHA,size,size))
    rowid = db.execute('SELECT rowid FROM models WHERE name=?', ('teacher',)).fetchone()[0]
    began = time.monotonic()
    with db.blobopen('models','data',rowid,readonly=False) as blob, open(filename,'rb') as src:
        count = 0
        while True:
            piece = src.read(1024*1024)
            if not piece:
                break
            blob.write(piece)
            count += len(piece)
            if count % (16*1024*1024) < len(piece):
                progress('Save teacher',count,size,began)
    progress('Save teacher',size,size,began)
    say()
    set_meta(db,'teacher_sha',TEACHER_SHA)
    db.commit()


def ensure_teacher(db):
    CACHE.mkdir(parents=True, exist_ok=True)
    if cached_valid(db):
        return GGUF
    row = db.execute('SELECT rowid,bytes,sha256 FROM models WHERE name=?', ('teacher',)).fetchone()
    if row and row[2] == TEACHER_SHA:
        say('Restoring teacher from BRAIN.db to system cache...')
        with db.blobopen('models','data',row[0],readonly=True) as src, GGUF.open('wb') as dst:
            sha = hashlib.sha256()
            while True:
                data = src.read(1024*1024)
                if not data:
                    break
                sha.update(data)
                dst.write(data)
        if sha.hexdigest() != TEACHER_SHA:
            GGUF.unlink(missing_ok=True)
            raise RuntimeError('Stored GGUF SHA256 mismatch; database may be corrupted')
        return GGUF
    say('Downloading Qwen2.5-0.5B-Instruct (about 491 MB, first run only)...')
    download(TEACHER_URL, GGUF, MAX_MODEL_BYTES, 'Qwen teacher')
    sha = hashlib.sha256()
    with GGUF.open('rb') as src:
        for chunk in iter(lambda:src.read(4*1024*1024),b''):
            sha.update(chunk)
    if sha.hexdigest() != TEACHER_SHA:
        GGUF.unlink(missing_ok=True)
        raise RuntimeError('Downloaded GGUF SHA256 mismatch. Check internet connection and retry.')
    write_model_to_db(db,GGUF)
    return GGUF


def server_binary():
    CACHE.mkdir(parents=True,exist_ok=True)
    suffix = '.exe' if os.name == 'nt' else ''
    found = list(CACHE.rglob('llama-server'+suffix))
    if found:
        return found[0]
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == 'windows' and machine in ('amd64','x86_64'):
        preferred = ('bin-win-cpu-x64.zip',)
    elif system == 'linux' and machine in ('x86_64','amd64'):
        preferred = ('bin-ubuntu-x64.tar.gz',)
    elif system == 'darwin':
        preferred = ('bin-macos-arm64.tar.gz',) if machine == 'arm64' else ('bin-macos-x64.tar.gz',)
    else:
        raise RuntimeError('Auto runtime supports Windows x64, Ubuntu x64, macOS arm64/x64; unsupported platform.')
    say('Installing official llama.cpp CPU runtime into user cache (not project folder)...')
    api = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=16'
    req = urllib.request.Request(api,headers={'User-Agent': USER_AGENT, 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as src:
        releases = json.load(src)
    assets = [(a['name'],a['browser_download_url']) for release in releases for a in release.get('assets',[])]
    match = next(((n,u) for n,u in assets if any(n.endswith(w) for w in preferred)),None)
    if not match:
        raise RuntimeError('Cannot find an official CPU llama.cpp release. Visit https://github.com/ggml-org/llama.cpp/releases')
    name,url = match
    archive = CACHE / name
    download(url,archive,250*1024*1024,'llama.cpp runtime')
    destination = CACHE / 'runtime'
    destination.mkdir(exist_ok=True)
    # Extract only normal files and defend against traversal.
    def safe_write(member_name, fileobj):
        path = (destination / member_name).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise RuntimeError('Unsafe llama.cpp archive path')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as out:
            shutil.copyfileobj(fileobj,out)
    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.is_dir():
                    if info.file_size > 250*1024*1024: raise RuntimeError('Unexpected runtime file size')
                    with z.open(info) as inp: safe_write(info.filename,inp)
    else:
        with tarfile.open(archive,'r:gz') as t:
            for member in t:
                if member.isfile():
                    if member.size > 250*1024*1024: raise RuntimeError('Unexpected runtime file size')
                    with t.extractfile(member) as inp: safe_write(member.name,inp)
    archive.unlink(missing_ok=True)
    found=list(destination.rglob('llama-server'+suffix))
    if not found:
        raise RuntimeError('llama-server missing from official llama.cpp archive')
    server=found[0]
    if os.name != 'nt': server.chmod(server.stat().st_mode | 0o111)
    return server


class Teacher:
    def __init__(self, model, parallel=4, role="Teacher"):
        self.model = model
        self.role = role
        self.parallel = max(1, min(12, int(parallel)))
        self.proc = None
        self.port = None
        self.log = None

    def __enter__(self):
        exe = server_binary()
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1',0))
            self.port = s.getsockname()[1]
        self.log = tempfile.TemporaryFile(mode='w+b')
        args=[str(exe), '-m',str(self.model),'--host','127.0.0.1','--port',str(self.port),
              '-c',str(self.parallel*2048),'-np',str(self.parallel),'-cb',
              '-t',str(min(8,max(2,(os.cpu_count() or 4)-1))),'-ngl','0']
        say(f'Starting local {self.role} model (CPU, {self.parallel} parallel slots)...')
        self.proc=subprocess.Popen(args,stdout=self.log,stderr=subprocess.STDOUT,cwd=str(exe.parent))
        limit=time.monotonic()+150
        while time.monotonic()<limit:
            if self.proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health',timeout=1) as res:
                    if res.status==200:
                        say(f'{self.role} model ready.')
                        return self
            except (OSError,urllib.error.URLError):
                time.sleep(.5)
        self.log.seek(0)
        logs=self.log.read()[-3000:].decode('utf-8','replace')
        self.__exit__(None,None,None)
        raise RuntimeError('Could not start llama-server. Logs:\n'+logs)

    def complete(self, messages, max_tokens=180, stream=False, temperature=.6):
        payload={'model':'local','messages':messages,'max_tokens':max_tokens,
                 'temperature':temperature,'stream':stream, 'top_p':0.9, 'repeat_penalty':1.12, 'frequency_penalty':0.15}
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',
            headers={'Content-Type':'application/json'},data=json.dumps(payload).encode())
        with urllib.request.urlopen(req,timeout=300) as response:
            if not stream:
                return json.load(response)['choices'][0]['message']['content']
            chunks=[]
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                data=raw[6:].strip()
                if data==b'[DONE]': break
                try:
                    delta=json.loads(data)['choices'][0].get('delta',{})
                    part=delta.get('content') or ''
                except (ValueError,KeyError,IndexError):
                    part=''
                if part:
                    print(part,end='',flush=True)
                    chunks.append(part)
            say()
            return ''.join(chunks)

    def __exit__(self,*args):
        if self.proc:
            self.proc.terminate()
            try: self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired: self.proc.kill()
        if self.log: self.log.close()


def example_lines(lines_limit=None):
    with DATA_PATH.open('r',encoding='utf-8',errors='replace') as file:
        for index,line in enumerate(file):
            if lines_limit is not None and index >= lines_limit: break
            try:
                q,a=line.rstrip('\r\n').split('\t',1)
                if q and a:
                    yield q,a
            except ValueError:
                continue


# === AXON student: original trainable neural word predictor (not a transformer) ===
# The network uses learned word embeddings and a position-weighted context, then
# scores candidate next words. The two teacher personas are NEVER used by AXON chat.
CONTEXT = 16
LOGIT_SCALE = 18.0
# Reserve five model input slots for the most recent human question.
# Keep a short sliding answer window in the remaining eleven slots.
POSITION = tuple([3.0] * 5 + [1.10 ** i for i in range(CONTEXT - 5)])
STUDENT_ARCH = 'AXON-human-assistant-curated-v5'
SPECIAL = ('<pad>', '<unk>', '<eos>', '<user>', '<assistant>')

# Natural-language first: not 30 million repetitive arithmetic samples.
# These examples contain no hardcoded output rule; they provide gradient targets.
SEED_PAIRS = [
    ('hi', 'Hello! How are you doing today?'),
    ('hello', 'Hello! What would you like to talk about?'),
    ('hey', 'Hey there! How is your day going?'),
    ('good morning', 'Good morning! I hope your day is going well.'),
    ('how are you', 'I am doing well. Thanks for asking!'),
    ('what is your name', 'My name is AXON. I am learning to have conversations.'),
    ('who are you', 'I am an experimental AI student that learns from text.'),
    ('what can you do', 'I can try to answer questions and learn from examples.'),
    ('how does learning work', 'Learning means adjusting model weights using examples and errors.'),
    ('what is artificial intelligence', 'Artificial intelligence is software trained to recognize and generate patterns.'),
    ('what are you learning', 'I am learning word relationships from conversations between humans and assistants.'),
    ('what is a computer', 'A computer is a machine that processes information using instructions.'),
    ('what is Python', 'Python is a programming language used for many kinds of software.'),
    ('how do I learn Python', 'Start with variables and loops, then practice with small programs.'),
    ('what is the weather', 'I do not have live weather information unless it is provided.'),
    ('tell me a joke', 'Why did the computer go to the doctor? It had a virus.'),
    ('thank you', 'You are welcome! I am happy to help.'),
    ('thanks', 'You are welcome!'),
    ('goodbye', 'Goodbye! Have a nice day.'),
    ('what makes a good conversation', 'A good conversation has clear questions, useful answers, and listening.'),
    ('why is water important', 'Water supports life and helps plants and animals survive.'),
    ('what is the sky', 'The sky is the atmosphere we see above the ground.'),
    ('do you understand me', 'I am learning language patterns, but I can still make mistakes.'),
    ('can you help me', 'I can try. Please tell me what you need.'),
    ('what should I do', 'Tell me more about your goal so I can give a useful suggestion.'),
    ('what is a book', 'A book is a collection of written pages that shares information or stories.'),
    ('what is music', 'Music combines sounds, rhythm, and melody to create expression.'),
    ('how can I study better', 'Study a little each day and practice recalling what you learned.'),
    ('explain machine learning', 'A machine learning model adjusts its parameters to predict patterns from examples.'),
    ('why do models make mistakes', 'Models can make mistakes when training examples are limited or misleading.'),
    ('how does a conversation work', 'One person speaks and the other responds with a relevant thought or question.'),
    ('what is a neural network', 'A neural network learns mathematical weights that transform inputs into outputs.'),
]
# Each session starts with a practical situation, not a free-form AI-to-AI debate.
# The human persona writes its own opener and follow-up questions.
SCENARIOS = [
    'You have just met a helpful assistant. Say hello and ask what it can help with.',
    'You want to learn a simple Python concept. Ask about variables or loops.',
    'You are practicing English. Ask how to improve your speaking confidence.',
    'You are studying for a school test. Ask for a practical study method.',
    'You want to understand why the sky is blue. Ask an ordinary science question.',
    'You want to learn how to write a short story. Ask for a starting idea.',
    'You are curious about how a computer works. Ask one clear question.',
    'You are troubleshooting a program that crashes. Ask how to debug it.',
    'You want to make a simple game. Ask for a first programming step.',
    'You are learning basic multiplication. Ask for an explanation or example.',
    'You are planning your day. Ask how to organize school and free time.',
    'You want to cook something simple. Ask for an easy recipe idea.',
    'You are wondering what causes rain. Ask for a beginner-friendly explanation.',
    'You are interested in space. Ask about the Moon or planets.',
    'You want to read more books. Ask how to choose a beginner-friendly book.',
    'You are learning about machine learning. Ask how models improve.',
    'You are bored and want something creative to do indoors. Ask for ideas.',
    'You want to learn basic music theory. Ask about rhythm or melody.',
    'You are confused by fractions. Ask for a short explanation.',
    'You want to create a website. Ask what HTML and CSS do.',
    'You want to understand the difference between RAM and storage.',
    'You are starting a new hobby. Ask how to practice consistently.',
    'You are curious why plants need sunlight. Ask a natural question.',
    'You want to improve your writing. Ask how to make sentences clearer.',
    'You want to plan a short workout. Ask for general beginner tips.',
    'You want to learn about electricity. Ask what a circuit is.',
    'You want to compare cats and dogs as pets. Ask an everyday question.',
    'You need to send a polite message. Ask how to phrase it.',
    'You heard the term artificial intelligence and want a simple definition.',
    'You want to know why sleep matters. Ask an everyday question.',
    'You are interested in taking better photographs. Ask a beginner question.',
    'You want to understand what a database does. Ask a simple question.',
    'You want to know how airplanes fly. Ask for a simple explanation.',
    'You want to learn to draw. Ask where to begin.',
    'You are learning geography. Ask about continents or oceans.',
    'You are trying to remember vocabulary. Ask for useful memory methods.',
    'You want to learn what causes a rainbow. Ask an everyday question.',
    'You want to become better at conversations. Ask how to ask follow-ups.',
    'You are interested in robots. Ask how sensors help them move.',
    'You want to understand what a computer virus is. Ask a simple question.',
]


def migrate_dialogue_schema(db):
    """Exclude random old Messenger/Receiver lessons from future training.

    Old notes remain preserved. On first upgrade, token weights are archived
    by build_vocabulary's student-architecture migration before new training.
    """
    if get_meta(db, 'human_assistant_migration') == '1' and get_meta(db, 'quality_gate_v5') == '1':
        return
    old = db.execute("SELECT COUNT(*) FROM lessons WHERE origin='dialogue'").fetchone()[0]
    if old:
        say(f'Archiving {old:,} previous random agent-to-agent examples (not training on them).')
        db.execute("UPDATE lessons SET origin='legacy_dialogue' WHERE origin='dialogue'")
    db.execute('DELETE FROM nextword')
    db.execute("DELETE FROM meta WHERE key='ngrams_ready'")
    set_meta(db, 'human_assistant_migration', '1')
    set_meta(db, 'quality_gate_v5', '1')
    db.commit()


def observe_word_patterns(db,text,commit=False,prefix=None):
    """Learn next-token counts conditioned on the question that prompted the answer.

    With prefix, the token stream is question + assistant marker + answer.
    This lets generation score likely first words from the current question,
    rather than using one global distribution of answer openings.
    """
    tokens=tokenizer(text)[:500]
    if not tokens:return
    if prefix:
        prev=['<user>']+tokenizer(prefix)[-120:]+['<assistant>']
    else:
        prev=['<start>']
    for token in tokens+['<eos>']:
        for n in (1,2):
            ctx=prev[-n:]
            if len(ctx)<n:continue
            key=json.dumps(ctx,ensure_ascii=False,separators=(',',':'))
            db.execute('INSERT INTO nextword(context,token,count) VALUES (?,?,1) '
                'ON CONFLICT(context,token) DO UPDATE SET count=count+1',(key,token))
        prev.append(token)
    if commit:db.commit()


def bootstrap_word_patterns(db):
    if get_meta(db,'ngrams_ready')=='2':return
    say('Preparing question-conditioned word-order statistics from saved conversations...')
    db.execute('DELETE FROM nextword')
    rows=db.execute(
        "SELECT prompt,answer FROM lessons "
        "WHERE origin IN ('seed','user','teacher','dialogue_quality_v5') "
        "ORDER BY id DESC LIMIT 12000"
    ).fetchall()
    for prompt,answer in rows:
        observe_word_patterns(db,answer,prefix=prompt)
    set_meta(db,'ngrams_ready','2')
    db.commit()


def word_pattern_probabilities(db,previous_tokens,lookup):
    """Return next-word probabilities learned from prior model conversations."""
    contexts=[]
    if len(previous_tokens)>=2:
        contexts.append(json.dumps(previous_tokens[-2:],ensure_ascii=False,separators=(',',':')))
    if previous_tokens:
        contexts.append(json.dumps(previous_tokens[-1:],ensure_ascii=False,separators=(',',':')))
    else:
        contexts.append(json.dumps(['<start>']))
    for ctx in contexts:
        rows=db.execute('SELECT token,count FROM nextword WHERE context=? ORDER BY count DESC LIMIT 120',(ctx,)).fetchall()
        selected=[(lookup[tok],int(count)) for tok,count in rows if tok in lookup]
        if selected:
            return selected
    return []


def bootstrap_lessons(db):
    migrate_dialogue_schema(db)
    if get_meta(db, 'seed_version') == '3':
        bootstrap_word_patterns(db)
        return
    for q, a in SEED_PAIRS:
        store_lesson(db, q, a, 'seed')
    set_meta(db, 'seed_version', '3')
    db.commit()
    bootstrap_word_patterns(db)


def build_vocabulary(db, max_lines=2000):
    """Build vocabulary from observed language, reserving room for future words."""
    if get_meta(db,'student_arch') != STUDENT_ARCH:
        if db.execute("SELECT 1 FROM weights WHERE key='emb_in'").fetchone():
            say('Preserving old student weights in BRAIN.db before upgrading its tokenizer.')
            enforce_db_size(db,160*1024*1024)
            for key in ('emb_in','emb_out'):
                db.execute("INSERT OR IGNORE INTO weights(key,value) SELECT ?,value FROM weights WHERE key=?",
                           ('legacy_v5_'+key,key))
            previous_vocab=get_meta(db,'vocab_json')
            if previous_vocab:
                set_meta(db,'legacy_v5_vocab',previous_vocab)
        db.execute("DELETE FROM weights WHERE key IN ('emb_in','emb_out')")
        db.execute("DELETE FROM meta WHERE key IN ('vocab_json','student_steps')")
        set_meta(db, 'student_arch', STUDENT_ARCH)
        db.commit()
    cached = get_meta(db, 'vocab_json')
    if cached:
        return json.loads(cached)
    bootstrap_lessons(db)
    say('Building tokenizer from real conversations; keeping space for new words...')
    priority = collections.Counter()
    for q, a in db.execute("SELECT prompt,answer FROM lessons WHERE origin IN ('seed','teacher','dialogue_quality_v5','user') ORDER BY id DESC LIMIT 15000"):
        priority.update(tokenizer(q+' '+a))
    general = collections.Counter()
    if DATA_PATH.is_file():
        for q,a in example_lines(max_lines):
            general.update(tokenizer(q+' '+a))
    vocab=list(SPECIAL)
    seen=set(vocab)
    for token,_ in priority.most_common(7500):
        if token not in seen and len(vocab)<8500:
            vocab.append(token);seen.add(token)
    # The mass-generated arithmetic corpus must not consume all vocab slots.
    for token,_ in general.most_common(1000):
        if token not in seen and len(vocab)<8500:
            vocab.append(token);seen.add(token)
    set_meta(db, 'vocab_json', json.dumps(vocab,ensure_ascii=False))
    db.commit()
    say(f'Vocabulary: {len(vocab):,} known words, {VOCAB_SIZE-len(vocab):,} free slots.')
    return vocab


def extend_vocabulary(db, text):
    vocab=build_vocabulary(db)
    seen=set(vocab)
    for word in tokenizer(text):
        if len(vocab)>=VOCAB_SIZE: break
        if word not in seen:
            seen.add(word);vocab.append(word)
    set_meta(db,'vocab_json',json.dumps(vocab,ensure_ascii=False))
    db.commit()
    return vocab


def note_exchange(db, session, turn, speaker, receiver, replied_to, message):
    enforce_db_size(db, 8*1024*1024)
    db.execute('INSERT INTO observations (session,turn,speaker,receiver,replied_to,message,created) VALUES (?,?,?,?,?,?,?)',
        (session,turn,speaker,receiver,replied_to[:2500],message[:2500],int(time.time())))
    # Preserve both sides of the chat, but learn the ANSWER task only from
    # replies by Assistant to Human, never from fake-human prompts.
    if speaker=='Assistant' and receiver=='Human' and replied_to.strip():
        store_lesson(db,replied_to,message,'dialogue_quality_v5')
        observe_word_patterns(db,message,prefix=replied_to)
    extend_vocabulary(db,replied_to+' '+message)
    db.commit()


def recent_notes(db, limit=8):
    return db.execute("SELECT speaker,receiver,message FROM observations WHERE speaker IN ('Human','Assistant') ORDER BY id DESC LIMIT ?",
                      (max(1,min(int(limit),100)),)).fetchall()[::-1]


def load_student(db, np, vocab):
    a = db.execute("SELECT value FROM weights WHERE key='emb_in'").fetchone()
    b = db.execute("SELECT value FROM weights WHERE key='emb_out'").fetchone()
    if a and b:
        def restore(blob):
            return np.frombuffer(zlib.decompress(blob), dtype='<f2').astype('float32').reshape(VOCAB_SIZE, DIM)
        return restore(a[0]), restore(b[0])
    say(f'Creating AXON neural student with {PARAMETERS:,} trainable weights...')
    rng = np.random.default_rng(2026)
    w_in = rng.normal(0, .025, (VOCAB_SIZE,DIM)).astype('float32')
    w_out = rng.normal(0, .025, (VOCAB_SIZE,DIM)).astype('float32')
    save_student(db, np, w_in, w_out, 0)
    return w_in,w_out


def save_student(db, np, w_in, w_out, steps):
    enforce_db_size(db, 160*1024*1024)
    for key,weights in (('emb_in',w_in),('emb_out',w_out)):
        value = zlib.compress(weights.astype('<f2').tobytes(), 1)
        db.execute('INSERT OR REPLACE INTO weights(key,value) VALUES (?,?)', (key,sqlite3.Binary(value)))
    set_meta(db,'student_steps',steps)
    db.commit()


def _sequence(q, a, lookup):
    prompt = [3] + [lookup.get(w,1) for w in tokenizer(q)] + [4]
    answer = [lookup.get(w,1) for w in tokenizer(a)] + [2]
    return prompt, answer


def _context(prefix):
    """Keep the current HUMAN question visible as the answer grows.

    The old window lost the user question after 16 generated tokens, making
    replies drift to random subjects. This fixed-size representation adds
    five anchored question tokens alongside eleven recent answer tokens.
    """
    try:
        answer_start=len(prefix)-1-prefix[::-1].index(4)
        user_start=max(i for i in range(answer_start) if prefix[i]==3)
        question=prefix[user_start+1:answer_start]
    except (ValueError, IndexError):
        question=[]
    if question:
        # Evenly sample question tokens, including its beginning and end.
        anchors=[3]+[question[round(i*(len(question)-1)/3)] for i in range(4)]
    else:
        anchors=[3,0,0,0,0]
    recent=([0]*11+prefix[-11:])[-11:]
    return anchors+recent


def _sparse_add(np, table, row_ids, row_grads):
    """Vectorized sum of repeated sparse embedding gradients (faster than np.add.at)."""
    order = np.argsort(row_ids,kind='stable')
    sorted_ids = row_ids[order]
    start = np.concatenate((np.array([0],dtype=np.int64),
                            np.nonzero(sorted_ids[1:]!=sorted_ids[:-1])[0]+1))
    grouped_grads = np.add.reduceat(row_grads[order],start,axis=0)
    table[sorted_ids[start]] += grouped_grads


def train_student(db, np, rows, max_steps=2000, print_every=100):
    vocab = build_vocabulary(db)
    lookup = {w:i for i,w in enumerate(vocab)}
    active = len(vocab)
    if active < 8:
        raise RuntimeError('Vocabulary is too small for training')
    w_in,w_out = load_student(db,np,vocab)
    rng = np.random.default_rng(2026 + int(get_meta(db, 'student_steps','0')))
    weights = np.asarray(POSITION, dtype=np.float32)
    weights /= weights.sum()
    total_steps = int(get_meta(db,'student_steps','0'))
    batch_count = 0
    batch_contexts = []
    batch_targets = []
    started = time.monotonic()
    last_loss = 0.0

    def update(contexts,targets):
        nonlocal total_steps,batch_count,last_loss
        ids = np.asarray(contexts,dtype=np.int32)
        targets = np.asarray(targets,dtype=np.int32)
        n = len(ids)
        neg = rng.integers(2,active,size=(n,31),dtype=np.int32)
        # No artificial role/padding tokens in sampled negatives.
        mask = (neg==3) | (neg==4)
        neg[mask] = rng.integers(5,active,size=int(mask.sum()),dtype=np.int32)
        neg[:,0] = np.where(targets==2,5,2)  # explicitly train EOS boundary
        candidates = np.concatenate((targets[:,None],neg),axis=1)
        context = np.einsum('bld,l->bd',w_in[ids],weights,optimize=True)
        sampled = w_out[candidates]
        scores = LOGIT_SCALE*np.einsum('bd,bkd->bk',context,sampled,optimize=True)
        scores -= scores.max(axis=1,keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1,keepdims=True)
        last_loss = float(-np.log(np.maximum(probs[:,0],1e-9)).mean())
        probs[:,0] -= 1
        probs /= n
        d_context = LOGIT_SCALE*np.einsum('bk,bkd->bd',probs,sampled,optimize=True)
        d_out = LOGIT_SCALE*(probs[:,:,None]*context[:,None,:])
        lr = .065 / (1.0 + total_steps/12000.0)
        _sparse_add(np,w_out,candidates.reshape(-1),-lr*np.clip(d_out.reshape(-1,DIM),-1,1))
        for index,posweight in enumerate(weights):
            _sparse_add(np,w_in,ids[:,index],-lr*np.clip(d_context*posweight,-1,1))
        batch_count += 1
        total_steps += 1
        if batch_count%print_every==0:
            progress('Student gradients',batch_count,max_steps if max_steps else 0,started)
            print(f'  loss {last_loss:.2f}',end='',flush=True)
        if batch_count%1000==0:
            save_student(db,np,w_in,w_out,total_steps)

    try:
        for q,a in rows:
            prefix,answer = _sequence(q,a,lookup)
            if len(prefix) > 128:
                prefix = [3]+prefix[-127:]
            if len(answer) > 160:
                answer = answer[:159] + [2]
            # Train on every answer position: the model must learn to complete
            # whole responses, not just randomly selected words in the question.
            for target in answer:
                batch_contexts.append(_context(prefix))
                batch_targets.append(target)
                prefix.append(target)
                if len(batch_contexts) >= 48:
                    update(batch_contexts,batch_targets)
                    batch_contexts.clear()
                    batch_targets.clear()
                    if max_steps is not None and batch_count >= max_steps:
                        break
            if max_steps is not None and batch_count>=max_steps:
                break
        if batch_contexts and (max_steps is None or batch_count<max_steps):
            update(batch_contexts,batch_targets)
    except KeyboardInterrupt:
        say('\nInterrupted: saving trained student weights...')
        save_student(db,np,w_in,w_out,total_steps)
        raise
    finally:
        say()
    save_student(db,np,w_in,w_out,total_steps)
    say(f'Trained {batch_count:,} batches. Last sampled loss {last_loss:.3f}; total {total_steps:,} batches.')
    return batch_count


def _format_piece(token,previous):
    if not previous:
        return token[0].upper()+token[1:] if token else ''
    if token in ('.',',','!','?',':',';','%',')',']','}','\'','’'):
        return token
    if previous in ('(','[','{','\'','’','$','#'):
        return token
    return ' '+token


def student_generate(db,prompt,length=65,temperature=.72,top_k=22,on_piece=None,state=None,history=None):
    """AXON generates genuinely new words, using only its own trained weights.

    Token-level constraints block punctuation loops; no answers are retrieved
    from lessons and the pretrained agents are NOT consulted during a chat reply.
    """
    np=ensure_numpy()
    vocab=build_vocabulary(db)
    if not db.execute("SELECT 1 FROM weights WHERE key='emb_in'").fetchone():
        return '[Run LLM Train.py to train AXON first.]'
    w_in,w_out=state if state is not None else load_student(db,np,vocab)
    lookup={word:i for i,word in enumerate(vocab)}
    prefix=[3]+[lookup.get(word,1) for word in tokenizer(prompt)]+[4]
    if history:
        for user,assistant in history[-2:]:
            prefix=[3]+[lookup.get(t,1) for t in tokenizer(user)]+[4]+[lookup.get(t,1) for t in tokenizer(assistant)]+[2]+prefix
    pos=np.asarray(POSITION,dtype=np.float32)
    pos/=pos.sum()
    rng=np.random.default_rng()
    ids_generated=[]
    pieces=[]
    last_token=''
    punctuation={'.',',','!','?',':',';','\"','\'','(',')','-'}
    for step in range(max(1,int(length))):
        context=np.einsum('ld,l->d',w_in[np.asarray(_context(prefix),dtype=np.int32)],pos,optimize=True)
        scores=LOGIT_SCALE*(w_out[:len(vocab)]@context)
        scores[:2]=-1e9
        scores[3:5]=-1e9
        if step<5: scores[2]=-1e9
        # Avoid initial punctuation and repeated punctuation while keeping learned
        # word ordering stochastic (not fixed/canned text).
        if step==0:
            for t in punctuation:
                if t in lookup: scores[lookup[t]]=-1e9
        if last_token in punctuation:
            for t in punctuation:
                if t in lookup: scores[lookup[t]]=-1e9
        if step>2 and last_token in {'.','!','?'}:
            scores[2]+=2.3
        for idx in set(ids_generated[-20:]):
            if idx>=5: scores[idx]-=1.65
        # If the network cannot form a sentence, produce a short partial reply
        # instead of dozens of repetitive commas.
        top=max(2,min(int(top_k),len(vocab)))
        candidates=np.argpartition(scores,-top)[-top:].tolist()
        learned=word_pattern_probabilities(db,[vocab[idx] for idx in ids_generated],lookup)
        for idx,count in learned:
            if scores[idx]>-1e8 and idx not in candidates:
                candidates.append(idx)
        candidates=np.asarray(candidates,dtype=np.int32)
        logits=scores[candidates]/max(.20,float(temperature))
        neural_probs=np.exp(logits-logits.max());neural_probs/=neural_probs.sum()
        if learned:
            pattern=np.zeros(len(candidates),dtype=np.float64)
            position_by_id={int(tok_id):i for i,tok_id in enumerate(candidates)}
            for idx,count in learned:
                if idx in position_by_id and scores[idx]>-1e8:
                    pattern[position_by_id[idx]]=float(count)**.8
            if pattern.sum()>0:
                pattern/=pattern.sum()
                mix=min(.72,.30+.11*math.log1p(sum(count for _,count in learned)))
                neural_probs=(1-mix)*neural_probs+mix*pattern
                neural_probs/=neural_probs.sum()
        choice=int(rng.choice(candidates,p=neural_probs))
        if choice==2: break
        token=vocab[choice]
        piece=_format_piece(token,last_token)
        last_token=token
        pieces.append(piece)
        if on_piece is not None: on_piece(piece)
        prefix.append(choice)
        ids_generated.append(choice)
        if len(ids_generated)>=12 and ids_generated[-6:]==ids_generated[-12:-6]:
            break
        if step>=16 and token in {'.','!','?'} and rng.random()<.35:
            break
    return ''.join(pieces)


# A human model produces each human turn, guided by a concrete, useful goal.
# These are goals, NOT canned assistant answers. Keep each conversation focused.
CHAT_GOALS = [
 ('greetings', ('Hi! How are you doing today?', 'What kinds of questions can I ask you?', 'Could you help me practice a short conversation?', 'What is a polite way to end a chat?')),
 ('python basics', ('Hi! What is a variable in Python?', 'Can I change a variable after creating it?', 'What is the difference between a list and a variable?', 'Could you show me a small example using both?')),
 ('python loops', ('What does a for loop do in Python?', 'When would I use a while loop instead?', 'How do I stop a loop from running forever?', 'Can you explain break using a simple example?')),
 ('school studying', ('Hi! How can I study for a test without getting distracted?', 'What is active recall?', 'How should I review what I learned tomorrow?', 'How can I tell whether I really understand a topic?')),
 ('english practice', ('How can I improve my spoken English?', 'What should I do when I forget a word?', 'Can you suggest a five-minute daily practice routine?', 'How can I build confidence in a conversation?')),
 ('math fractions', ('Hi! What does one half mean?', 'How do I add one half and one quarter?', 'Why do fractions need a common denominator?', 'Can you give me a practice problem and explain it?')),
 ('simple multiplication', ('Why is six times seven equal to forty-two?', 'Is multiplication repeated addition?', 'How can I practice times tables?', 'Can you demonstrate a quick way to check the result?')),
 ('basic science', ('Why is the sky blue during the day?', 'Why can the sunset look orange or red?', 'Does the color change on cloudy days?', 'How is sunlight scattered in the atmosphere?')),
 ('rain and clouds', ('How do clouds form?', 'What makes a cloud start raining?', 'What is the difference between rain and snow?', 'Why do some clouds look darker than others?')),
 ('space', ('Hi! Why does the Moon change shape during the month?', 'Does the Moon produce its own light?', 'What causes a lunar eclipse?', 'How long does the Moon take to orbit Earth?')),
 ('plants', ('Why do plants need sunlight?', 'What do roots do for the plant?', 'How do leaves help a plant grow?', 'What happens when a plant gets too little water?')),
 ('writing a story', ('Hi! How do I start writing a short story?', 'How can I make a character interesting?', 'What is a simple way to create conflict?', 'How should a short story end?')),
 ('writing clearly', ('How can I write clearer sentences?', 'When should I start a new paragraph?', 'How can I remove repeated words?', 'Could you show a before-and-after example?')),
 ('computers', ('What does a CPU actually do?', 'How is RAM different from storage?', 'Why does a computer slow down with many apps open?', 'What are some safe ways to make it faster?')),
 ('websites', ('Hi! What do HTML and CSS do?', 'How does JavaScript change a webpage?', 'Can I build a small website with just those languages?', 'What would be a good first webpage to practice?')),
 ('debugging', ('My Python script crashes. What should I check first?', 'How do I read a Python error traceback?', 'What is a useful way to test one part of the code?', 'How do I know whether the bug is actually fixed?')),
 ('games', ('How can I start making a simple game?', 'What is a game loop?', 'How should I handle keyboard input?', 'What small feature should I build first?')),
 ('robots', ('How does a robot detect obstacles?', 'What is the difference between a sensor and a motor?', 'How can a robot decide when to turn?', 'Why might a robot get stuck while moving?')),
 ('databases', ('What is a database used for?', 'What is the difference between a table and a row?', 'Why would an app use SQLite?', 'How does a program find a stored record?')),
 ('machine learning', ('What does it mean to train an AI model?', 'What are model parameters?', 'Why can a model memorize instead of generalizing?', 'What can improve the quality of training examples?')),
 ('music', ('Hi! What is rhythm in music?', 'How is melody different from rhythm?', 'What does tempo mean?', 'How can a beginner practice keeping a steady beat?')),
 ('photography', ('How can I take sharper pictures with a camera?', 'What is shutter speed?', 'Why do indoor photos sometimes look blurry?', 'When should I use a tripod?')),
 ('cooking', ('What is an easy meal a beginner can cook?', 'What ingredients should I prepare first?', 'How can I avoid overcooking the food?', 'How should I store leftovers safely?')),
 ('exercise', ('How can I begin exercising at home?', 'Why is warming up useful?', 'How can I increase difficulty gradually?', 'Why do rest days matter?')),
 ('planning', ('How can I organize my homework and free time?', 'What should I do when a task feels too big?', 'How can I decide which task to do first?', 'How do I review my plan at the end of the day?')),
 ('books', ('How can I choose a book that is not too difficult?', 'What should I do if I lose focus while reading?', 'How can I remember important things from a book?', 'What are some ways to build a reading habit?')),
 ('memory', ('Why do I forget things I recently learned?', 'What is spaced repetition?', 'How often should I review new vocabulary?', 'How can I test myself without looking at notes?')),
 ('geography', ('What is the difference between a continent and a country?', 'How many continents are commonly taught?', 'What is an ocean current?', 'Why do maps sometimes distort the sizes of countries?')),
 ('electricity', ('What is an electric circuit?', 'Why does a circuit need a closed path?', 'What is the difference between voltage and current?', 'What does a resistor do?')),
 ('animals', ('Why do cats purr?', 'How do dogs communicate with body language?', 'Why do pets need regular exercise?', 'What should a new pet owner prepare?')),
 ('drawing', ('How can I learn to draw if I am a beginner?', 'Why should I practice simple shapes?', 'How can I make a drawing look three-dimensional?', 'What is a good daily drawing exercise?')),
 ('airplanes', ('How does an airplane stay in the air?', 'What do the wings do?', 'Why are airplane engines needed?', 'How does a pilot control a turn?')),
 ('sleep', ('Why is getting enough sleep important?', 'How can I make a regular sleep schedule?', 'Why can screens make it harder to sleep?', 'What is a good way to unwind before bed?')),
 ('weather', ('What causes wind?', 'Why does warm air rise?', 'How are weather forecasts made?', 'Why can the weather change quickly?')),
 ('friendship', ('Hi! How can I start a conversation with a new classmate?', 'How can I ask a good follow-up question?', 'What should I say when we disagree politely?', 'How do I keep a conversation from becoming one-sided?')),
 ('language learning', ('How should I start learning a new language?', 'What is a useful way to practice new words?', 'Should I focus on speaking or grammar first?', 'How can I measure my progress each month?')),
 ('internet safety', ('How do I make a strong password?', 'Why should I avoid clicking suspicious links?', 'What is two-factor authentication?', 'What should I do if I think an account was compromised?')),
 ('science experiments', ('What is a scientific hypothesis?', 'Why should an experiment change only one variable?', 'What is a control group?', 'How should I record experimental results?')),
 ('computer storage', ('What is the difference between an SSD and an HDD?', 'Why is an SSD usually faster?', 'What should I back up before replacing a drive?', 'How do I estimate how much storage I need?')),
 ('everyday questions', ('Hi! What can I do when I am bored at home?', 'What is a creative activity that needs few supplies?', 'How can I turn it into a small project?', 'How can I share what I made with a friend?')),
]


# Many distinct subject/aspect categories. Every combination is a separate
# objective for a real local-Human/local-Assistant teacher dialogue.
LEARNING_SUBJECTS = "algebra|geometry|calculus|statistics|probability|trigonometry|logic|number theory|financial mathematics|physics|mechanics|optics|acoustics|thermodynamics|electricity|magnetism|astronomy|astrophysics|cosmology|chemistry|organic chemistry|biochemistry|materials science|biology|genetics|evolution|microbiology|botany|zoology|anatomy|physiology|neuroscience|immunology|nutrition|public health|veterinary science|geology|earthquakes|volcanoes|weather|meteorology|climatology|oceanography|hydrology|forestry|ecology|conservation|sustainability|biodiversity|soil science|agriculture|gardening|mammals|birds|reptiles|amphibians|insects|fish|marine animals|forest ecosystems|wetlands|rivers|mountains|deserts|rainforests|seasons|water cycle|world history|ancient civilizations|medieval history|modern history|archaeology|anthropology|world geography|cartography|geopolitics|civics|philosophy|ethics|sociology|psychology|linguistics|literature|grammar|writing|creative writing|public speaking|debate|reading comprehension|English language|Spanish language|French language|Turkish language|Portuguese language|translation|storytelling|poetry|Python programming|JavaScript programming|HTML|CSS|Java programming|C programming|C++ programming|Rust programming|SQL|algorithms|data structures|software debugging|web development|mobile apps|game development|computer graphics|operating systems|computer networking|cybersecurity|privacy|cloud computing|databases|machine learning|artificial intelligence|robotics|embedded electronics|computer vision|natural language processing|large language models|information retrieval|data analysis|data visualization|software testing|version control|open source software|computer hardware|microcontrollers|sensors|motors|automation|electrical engineering|mechanical engineering|civil engineering|chemical engineering|aerospace engineering|industrial engineering|architecture|construction|carpentry|plumbing|welding|HVAC|automotive repair|bicycle repair|electronics repair|medicine|nursing|dentistry|pharmacy|emergency response|first aid|occupational safety|healthcare administration|physical therapy|fitness|exercise science|sleep science|economics|personal finance|budgeting|accounting|business management|entrepreneurship|marketing|sales|customer service|human resources|project management|supply chains|logistics|warehouse operations|retail work|office administration|law basics|government|education|teaching|study skills|research methods|journalism|media literacy|library science|information science|career planning|job applications|job interviews|workplace communication|teamwork|leadership|negotiation|cooking|baking|food science|food safety|household maintenance|home organization|sewing|textiles|fashion design|interior design|photography|filmmaking|animation|graphic design|drawing|painting|sculpture|music theory|music production|singing|dance|theater|sports|football|basketball|tennis|swimming|cycling|hiking|camping|navigation|travel planning|transportation|railways|aviation|maritime navigation|urban planning|public transport|friendship|relationships|conversation skills|conflict resolution|time management|habit building|decision making|critical thinking|problem solving|creativity|personal development|digital citizenship|internet safety".split("|")
LEARNING_ASPECTS = (
    "fundamentals", "history", "core ideas", "applications",
    "beginner examples", "advanced concepts", "important tools",
    "common mistakes", "practical methods", "workplace examples",
    "nature connections", "comparisons", "simple experiments",
    "current research", "daily life", "safety", "ethics",
    "problem solving", "practice exercises", "future developments",
)
STYLE_MODES = (
    "plain", "casual", "slang", "professional", "curious", "friendly",
    "old-fashioned", "direct", "polite", "simple", "hello", "reflective",
)
def make_topic_scenarios():
    scenarios = list(CHAT_GOALS)
    for subject in LEARNING_SUBJECTS:
        for aspect in LEARNING_ASPECTS:
            scenarios.append((f"{subject}: {aspect}", (
                f"Could you explain {aspect} of {subject} in a beginner-friendly way?",
                f"What is one specific example of {aspect} in {subject}?",
                f"What mistakes should a learner avoid regarding {aspect} of {subject}?",
                f"How could I practice or apply {aspect} of {subject} myself?",
            )))
    return scenarios
ALL_SCENARIOS = make_topic_scenarios()

HUMAN_SYSTEM = (
    'You are a HUMAN customer or student texting an AI assistant. You are NOT the assistant. '
    'You need useful information, and you ask everyday questions like a real person. '
    'Write exactly ONE short first-person HUMAN message, max 28 words. '
    'Your message must ask the specific target question described in the user instruction. '
    'You may rephrase that question naturally but MUST keep the subject and meaning. '
    'If the previous assistant gave a helpful answer, acknowledge it briefly and ask the new target question. '
    'Do NOT answer the question. NEVER say: How can I assist you, How can I help you, '
    'I am here to help, What can I help you with, or I am a chatbot. '
    'Do NOT use slash-separated choices such as Hi/Hello. '
    'Examples of GOOD human messages: Hi! What is a Python variable? | '
    'Thanks. Can I change its value later? | Why is the sky blue? '
    'Reply with only the human message, no quotation marks or role labels.'
)

ASSISTANT_SYSTEM = (
    'You are a helpful tutor answering an actual human question. You are the ASSISTANT, not the human. '
    'Read the most recent human message carefully and answer its question directly, with concrete and correct information. '
    'Use 1 to 3 clear sentences. Prefer an example over generic filler. '
    'Do not start with Hello or Hi unless the human only greeted you. '
    'Never say How can I assist you today, I am here to help, or I am just a chatbot. '
    'Never repeat your previous answer. Do not change the topic. '
    'Do not pretend to be the human and do not produce speaker labels or a transcript.'
)

STOP_TOPIC_WORDS = set("a an and are as at be can could do does explain for from have how i in is it me my of on or our please should so that the them there these this to was what when where which who why will with would you your about tell give show simple small beginner more hello hi thanks thank".split())
ROLE_PHRASES = ('how can i assist you', 'how may i assist you', 'how can i help you',
                'what can i help you with', 'i am here to assist', 'i am here to help',
                'i am just a chatbot', "i'm just a chatbot", 'as an ai language model',
                'as a language model', 'i am a virtual assistant', "i'm an ai assistant")

def _normalize_message(value):
    return ' '.join(re.findall(r"[a-z0-9']+", (value or '').casefold()))


def _similarity(left,right):
    return difflib.SequenceMatcher(None,_normalize_message(left),_normalize_message(right)).ratio()


def _clean_message(value, speaker):
    value=re.sub(r'\s+', ' ', (value or '').strip())
    value=re.sub(r'^(human|user|assistant|ai|answer|message)\s*:\s*', '',value,flags=re.I).strip(' "\'')
    value=value[:520].strip()
    low=value.casefold()
    words=value.split()
    if not 4<=len(value)<=520 or len(words)<2:
        return ''
    if any(bad in low for bad in ('<|im_start|>','<|im_end|>','human:','assistant:',
                                    'system prompt:','user:','```','\ufffd')):
        return ''
    if len(set(tokenizer(value)))<max(2,len(tokenizer(value))//6):
        return ''
    if speaker=='Human':
        if len(words)>32 or any(phrase in low for phrase in ROLE_PHRASES):
            return ''
        if re.search(r'\b(hi|hello|hey)\s*/\s*(hi|hello|hey)\b',low):
            return ''
        if low.startswith(('sure,','certainly,','of course,','yes, i can','i would be happy')):
            return ''
    elif len(words)>110 or any(phrase in low for phrase in ROLE_PHRASES):
        return ''
    return value


def _target_relevance(message,target):
    keywords={w for w in _normalize_message(target).split() if w not in STOP_TOPIC_WORDS and len(w)>=4}
    if not keywords:
        return True
    found=set(_normalize_message(message).split())
    # A word overlap or lexical near match is enough; do not rely on the weak
    # 0.5B human model to produce a perfect paraphrase.
    return bool(found & keywords) or any(any(k.startswith(w[:5]) for k in keywords) for w in found if len(w)>=5)



def _human_request(goal, target, previous_answer, first_style, pair_index):
    voices = {
        "plain": "Ask naturally in everyday English, without a forced greeting.",
        "casual": "Sound relaxed and conversational, like a real person texting.",
        "slang": "Use light, modern slang but ask a clear, real question.",
        "professional": "Ask politely, like someone discussing a practical issue at work.",
        "curious": "Sound curious. Ask a focused question.",
        "friendly": "Use a friendly tone.",
        "old-fashioned": "Occasionally use mild old-fashioned English like Good day or pray tell.",
        "direct": "Ask briefly and directly.",
        "polite": "Be courteous but never offer the AI your assistance.",
        "simple": "Use simple words and short sentences.",
        "hello": "On the first turn only, start with Hi, Hello or Hey.",
        "reflective": "Ask a thoughtful, concrete question.",
    }
    voice = voices.get(first_style, voices["plain"])
    if pair_index > 0:
        voice += " No greeting, and do not repeat your last question."
    previous = ("Previous assistant answer: " + previous_answer[:230] + "\n") if previous_answer else ""
    return (
        ("Start a new human conversation.\n" if pair_index == 0 else "Ask a related follow-up question.\n")
        + "Your learning topic is: " + goal + ".\n" + previous
        + "TARGET QUESTION: " + target + "\nVOICE: " + voice + "\n"
        + "Write only ONE human question, maximum 28 words. "
        + "Do not answer it, offer to help the assistant or add role labels."
    )

def _pair_quality(human,answer,history):
    h=_normalize_message(human)
    a=_normalize_message(answer)
    if any(bad in h for bad in ROLE_PHRASES) or any(bad in a for bad in ROLE_PHRASES):
        return False,'role confusion / generic help offer'
    if h==a or _similarity(human,answer)>.85:
        return False,'assistant echoed human'
    if len(answer.split())<5 and len(human.split())>4:
        return False,'answer too short'
    if 'how can i assist' in a or 'what can i help' in a:
        return False,'unhelpful reply'
    if any(_similarity(human,old_h)>.84 for old_h,_ in history):
        return False,'repeated human question'
    if any(_similarity(answer,old_a)>.86 for _,old_a in history):
        return False,'repeated assistant answer'
    return True,'ok'


def _chat_worker(human_model, assistant_model, turns, scenario, first_style, session, notes_queue=None):
    # scenario is (topic, question objectives), NOT a vague prompt to improvise.
    goal, targets=scenario
    history=[]
    pairs=[]
    for pair_index in range(max(1,int(turns)//2)):
        if pair_index < len(targets):
            target=targets[pair_index]
        else:
            followups=(
                f'Could you give me a practical example of {goal} that we have not discussed?',
                f'What common beginner mistake should I avoid when learning {goal}?',
                f'How could I practice {goal} on my own today?',
                f'Can you give me a short question to test my understanding of {goal}?',
                f'What is an easy way to explain {goal} to a friend?',
                f'How would {goal} be useful in everyday life?',
                f'What is the next thing I should learn about {goal}?',
                f'Could you compare two different examples involving {goal}?',
            )
            target=followups[(pair_index-len(targets))%len(followups)]
            cycle=(pair_index-len(targets))//len(followups)
            if cycle:target=target.rstrip('?')+f' with a different example than before (practice {cycle+1})?'
        previous_answer=history[-1][1] if history else ''
        human=''; used_fallback=False
        for attempt in range(3):
            prompt=_human_request(goal,target,previous_answer,first_style,pair_index)
            if attempt:
                prompt+='\nIMPORTANT: The last candidate was invalid. Ask about '+target+' Do not sound like a chatbot assistant.'
            try:
                raw=human_model.complete([{'role':'system','content':HUMAN_SYSTEM},
                                          {'role':'user','content':prompt}],
                                         max_tokens=75,temperature=.78+.08*attempt)
            except Exception:
                raw=''
            candidate=_clean_message(raw,'Human')
            if not candidate or not _target_relevance(candidate,target):
                continue
            if not (candidate.endswith('?') or '?' in candidate or
                    candidate.casefold().startswith(('explain','tell me','show me','give me'))):
                continue
            if any(_similarity(candidate,past[0])>.82 for past in pairs):
                continue
            human=candidate
            break
        if not human:
            # Reliable last resort; never save a model's role-confused sentence.
            human=('Hi! '+target[0].lower()+target[1:] if pair_index==0 and first_style=='hello'
                   and not target.casefold().startswith(('hi','hello','hey')) else target)
            used_fallback=True
        answer=''
        messages=[{'role':'system','content':ASSISTANT_SYSTEM}]
        for old_h,old_a in pairs[-3:]:
            messages.extend(({'role':'user','content':old_h},{'role':'assistant','content':old_a}))
        messages.append({'role':'user','content':human})
        for attempt in range(3):
            if attempt:
                messages[0]={'role':'system','content':ASSISTANT_SYSTEM+
                    ' The last attempt was too generic or repeated. Explain the key fact in this new question specifically.'}
            try:
                raw=assistant_model.complete(messages,max_tokens=150,temperature=.62+.09*attempt)
            except Exception:
                raw=''
            candidate=_clean_message(raw,'Assistant')
            valid,reason=_pair_quality(human,candidate,pairs) if candidate else (False,'empty')
            if valid:
                answer=candidate
                break
        if not answer:
            # Do not train on empty, repeated or generic assistant outputs.
            continue
        pairs.append((human,answer))
        history.append((human,answer))
        if notes_queue is not None:
            notes_queue.put((session,goal,pair_index,human,answer,
                             pairs[-2][1] if len(pairs)>1 else goal,used_fallback))
    return session,goal,pairs


def agent_dialogues_parallel(db, human_model, assistant_model=None, chats=4, turns=8, topic=None, display=True):
    """Run concurrent goal-directed human/assistant chats and filter BEFORE storage."""
    assistant_model=assistant_model or human_model
    chats=max(1,min(12,int(chats)))
    turns=max(2,int(turns))
    if turns%2:turns+=1
    seed=int(get_meta(db,'human_chat_sessions',0))
    pairs_saved=0;skipped=0;fallbacks=0
    accepted_sessions=set()
    recent_rows=db.execute("SELECT prompt,answer FROM lessons WHERE origin='dialogue_quality_v5' ORDER BY id DESC LIMIT 800").fetchall()
    recent=collections.deque(recent_rows,maxlen=900)
    if display:
        say(f'\nStarting {chats} independent, goal-directed Human/Assistant conversations.')
        say(f'Each session: up to {turns//2} useful human questions + assistant answers.')
        say('AXON accepts only distinct, valid Q&A pairs; rejected repetition is NOT learned.\n')
    notes_queue=queue.Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=chats,thread_name_prefix='axon_chat') as pool:
        futures=[]
        for index in range(chats):
            if topic:
                subject=topic.strip()[:100]
                scenario=(subject,(f'Could you explain {subject} in simple terms?',
                                   f'What is an everyday example involving {subject}?',
                                   f'What mistake should a beginner avoid with {subject}?',
                                   f'How could I practice understanding {subject}?'))
            else:
                scenario=ALL_SCENARIOS[(seed+index)%len(ALL_SCENARIOS)]
            session=uuid.uuid4().hex[:12]
            futures.append(pool.submit(_chat_worker,human_model,assistant_model,turns,scenario,
                                       STYLE_MODES[(seed+index)%len(STYLE_MODES)],session,notes_queue))
        while True:
            try:
                session,goal,index,human,assistant,previous,used_fallback=notes_queue.get(timeout=.3)
            except queue.Empty:
                if all(f.done() for f in futures):break
                continue
            # Cross-chat and cross-run duplicate check; avoid poisoning the DB.
            if any(_similarity(human,old_h)>.90 or
                   (len(assistant.split())<35 and _similarity(assistant,old_a)>.96)
                   for old_h,old_a in recent):
                skipped+=1
                if display:say(f'[Chat {session} | exchange {index+1}] SKIPPED repeated Q&A')
                continue
            note_exchange(db,session,2*index+1,'Human','Assistant',previous,human)
            note_exchange(db,session,2*index+2,'Assistant','Human',human,assistant)
            recent.append((human,assistant))
            pairs_saved+=1
            fallbacks+=int(used_fallback)
            accepted_sessions.add(session)
            if display:
                say(f'[Chat {session} | exchange {index+1}] Human > {human}')
                say(f'[Chat {session} | exchange {index+1}] Assistant > {assistant}')
                say(f'    AXON accepted Q&A #{pairs_saved:,} for training'+
                    (' (human fallback question)' if used_fallback else ''))
        for future in concurrent.futures.as_completed(futures):
            try:
                _,_,pairs=future.result()
                if not pairs:say('A chat produced no valid pairs; it was not used for training.')
            except Exception as exc:
                say(f'One chat failed and was skipped: {exc}')
    set_meta(db,'human_chat_sessions',seed+chats)
    db.commit()
    say(f'Completed {len(accepted_sessions)}/{chats} useful sessions: {pairs_saved} unique Q&A pairs; '
        f'{skipped} duplicates rejected; {fallbacks} fallback human questions.')
    return len(accepted_sessions),pairs_saved

def agent_dialogue(db, teacher, turns=8, topic=None, display=True):
    """Compatibility shim for older chat scripts."""
    sessions,pairs=agent_dialogues_parallel(db,teacher,chats=1,turns=turns,topic=topic,display=display)
    return str(sessions),pairs*2


def learn_observations(db,np,steps=250,epochs=25):
    """Replay teacher conversations to AXON as next-token gradient targets."""
    rows=db.execute("SELECT prompt,answer FROM lessons WHERE origin='dialogue_quality_v5' ORDER BY id DESC LIMIT 3000").fetchall()
    seeds=db.execute("SELECT prompt,answer FROM lessons WHERE origin IN ('seed','user','teacher') ORDER BY id DESC LIMIT 1500").fetchall()
    if not rows and not seeds:
        say('No observations to train; skipping.')
        return 0
    say(f'AXON learning from {len(rows):,} observed turns + {len(seeds):,} seed/user examples...')
    # Favor real conversations, while keeping common greetings learned.
    def training_rows():
        epoch=0
        while True:   # train for exactly the requested number of gradient steps
            pool=list(rows)+list(rows)+list(seeds)
            random.Random(2026+epoch).shuffle(pool)
            yield from pool
            epoch+=1
    return train_student(db,np,training_rows(),max_steps=max(1,int(steps)),print_every=25)


def main():
    parser=argparse.ArgumentParser(description='Parallel human/assistant conversations train AXON locally')
    parser.add_argument('--chats',type=int,default=4,help='simultaneous human/assistant chats (default: 4, max: 12)')
    parser.add_argument('--rounds',type=int,default=8,help='messages PER chat; even number, default: 8 (4 Q&A pairs)')
    parser.add_argument('--steps',type=int,default=400,help='student neural gradient batches per cycle (default: 400)')
    parser.add_argument('--topic',type=str,default=None,help='one practical human question topic for all chats')
    parser.add_argument('--forever',dest='forever',action='store_true',default=True,help='continue until Ctrl+C (default)')
    parser.add_argument('--once',dest='forever',action='store_false',help='only one cycle')
    parser.add_argument('--shared-model',action='store_true',help='use one local Qwen server for both roles (less memory, not separate processes)')
    parser.add_argument('--offline',action='store_true',help='train from existing notes and user lessons, no download')
    parser.add_argument('--corpus-examples',type=int,default=0,help='extra synthetic corpus rows (default: 0)')
    parser.add_argument('--all',action='store_true',help='also use entire synthetic corpus; very slow and not conversational')
    parser.add_argument('--distill',type=int,default=None,help='legacy alias for --rounds')
    parser.add_argument('--examples',type=int,default=None,help='legacy alias for --corpus-examples')
    args=parser.parse_args()
    if args.distill is not None:args.rounds=args.distill
    if args.examples is not None:args.corpus_examples=args.examples
    if not 1<=args.chats<=12:parser.error('--chats must be between 1 and 12')
    if not 2<=args.rounds<=200:parser.error('--rounds must be between 2 and 200')
    if args.steps<1:parser.error('--steps must be at least 1')
    if sys.version_info<(3,11):raise RuntimeError('Python 3.11 or newer is required.')
    say('AXON AI | continuous varied Human/Assistant conversations')
    say(f'{len(ALL_SCENARIOS):,} learning categories, {len(STYLE_MODES)} speaking styles; Ctrl+C to stop.')
    say(f'Student: {PARAMETERS:,} parameters. Maximum BRAIN.db: 15 GiB.')
    say(f'Human and Assistant use separate local Qwen processes by default, with {args.chats} slots each.')
    np=ensure_numpy()
    with connection() as db:
        bootstrap_lessons(db)
        if args.offline:
            say('Offline: learning from saved human/assistant Q&A pairs and seed lessons.')
            learn_observations(db,np,steps=args.steps)
        else:
            model=ensure_teacher(db)
            def training_loop(human_model,assistant_model):
                while True:
                    sessions,pairs=agent_dialogues_parallel(
                        db,human_model,assistant_model,chats=args.chats,turns=args.rounds,topic=args.topic)
                    if pairs:
                        say(f'\nAXON training from {pairs} new human-to-assistant answers...')
                        learn_observations(db,np,steps=args.steps)
                    else:
                        say('No valid answers generated, so no training occurred this cycle.')
                    enforce_db_size(db)
                    say(f'BRAIN.db: {DB_PATH.stat().st_size/1024**2:.1f} MiB / 15360 MiB.')
                    if not args.forever:break
            with Teacher(model,parallel=args.chats,role='Human') as human_model:
                if args.shared_model:
                    training_loop(human_model,human_model)
                else:
                    with Teacher(model,parallel=args.chats,role='Assistant') as assistant_model:
                        training_loop(human_model,assistant_model)
        if args.all or args.corpus_examples>0:
            if not DATA_PATH.is_file():say('No training_data.txt found; skipped corpus stage.')
            else:
                say('Extra synthetic math/programming corpus stage...')
                train_student(db,np,example_lines(None if args.all else max(0,args.corpus_examples)),
                              max_steps=None if args.all else max(1,args.steps))
        say('Done. Run LLM Chat.py to interact with AXON student word-by-word.')
        say('More learning: python "LLM Train.py" --chats 8 --rounds 10 --steps 1200')

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:say('\nStopped. Notes and last student checkpoint saved.')
    except Exception as exc:
        say('Training error: '+str(exc))
        raise