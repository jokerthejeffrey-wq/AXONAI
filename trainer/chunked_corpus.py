"""Fetch only assigned, SHA-256 verified corpus pages from the release CDN."""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen
from trainer.corpus import CORPUS_ID, REVISION, Progress

BASE = 'https://github.com/jokerthejeffrey-wq/AXONAI/releases/download/corpus-smoltalk-v1'
MANIFEST = Path(__file__).with_name('corpus-manifest.json')


def load_manifest(path=MANIFEST):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    info = data['corpus']
    if info['id'] != CORPUS_ID or info['revision'] != REVISION:
        raise ValueError('Unsupported corpus revision')
    offset = 0
    for page in data['pages']:
        if (page['start'] != offset or not 1 <= page['count'] <= 1024 or
            page['name'] != f'page-{offset:07d}.jsonl.gz' or
            not 0 < page['size'] <= 16*1024**2 or
            not 0 < page['raw_size'] <= 32*1024**2 or
            len(page['sha256']) != 64):
            raise ValueError('Invalid corpus page manifest')
        offset += page['count']
    if offset != info['records']:
        raise ValueError('Incomplete corpus page manifest')
    return data


def read_assigned(folder, manifest, start, count, opener=urlopen):
    if not isinstance(start, int) or not isinstance(count, int) or start < 0 or count < 1 or start+count > manifest['corpus']['records']:
        raise ValueError('Invalid assigned corpus range')
    folder = Path(folder) / 'pages'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in manifest['pages']:
        if page['start'] >= start+count or page['start']+page['count'] <= start:
            continue
        path = folder / page['name']
        valid = path.exists() and path.stat().st_size == page['size']
        if valid:
            valid = hashlib.sha256(path.read_bytes()).hexdigest() == page['sha256']
        if not valid:
            meter = Progress('Assigned lessons', page['size'])
            fd, temp = tempfile.mkstemp(dir=folder, suffix='.partial')
            try:
                digest = hashlib.sha256(); size = 0
                with os.fdopen(fd, 'wb') as out, opener(BASE+'/'+page['name'], timeout=60) as response:
                    while block := response.read(128*1024):
                        size += len(block)
                        if size > page['size']:
                            raise ValueError('Oversized corpus page')
                        out.write(block); digest.update(block); meter.update(size)
                if size != page['size'] or digest.hexdigest() != page['sha256']:
                    raise ValueError('Corpus page checksum mismatch; reopen to retry')
                os.replace(temp, path)
                meter.update(size, force=True)
            finally:
                Path(temp).unlink(missing_ok=True)
                meter.finish()
        with gzip.open(path, 'rb') as source:
            raw = source.read(page['raw_size']+1)
        if len(raw) != page['raw_size']:
            raise ValueError('Invalid expanded corpus page size')
        records = raw.splitlines()
        if len(records) != page['count']:
            raise ValueError('Invalid corpus page count')
        a = max(start-page['start'], 0)
        b = min(start+count-page['start'], len(records))
        rows.extend(json.loads(line) for line in records[a:b])
    if len(rows) != count:
        raise ValueError('Incomplete assigned data')
    return rows
