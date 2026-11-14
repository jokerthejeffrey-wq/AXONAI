"""Build the canonical corpus ONCE on CI, not on every volunteer PC."""
import gzip
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trainer.corpus import prepare


def package(cache, output, target_manifest):
    info = prepare(cache)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pages = []; start = 0
    with (Path(cache)/'training.txt').open('rb') as source:
        while True:
            lines = []
            for _ in range(1024):
                line = source.readline()
                if not line: break
                lines.append(line)
            if not lines: break
            raw = b''.join(lines)
            compressed = gzip.compress(raw, compresslevel=6, mtime=0)
            name = f'page-{start:07d}.jsonl.gz'
            (output/name).write_bytes(compressed)
            pages.append(dict(start=start,count=len(lines),name=name,size=len(compressed),
                              raw_size=len(raw),sha256=hashlib.sha256(compressed).hexdigest()))
            start += len(lines)
    if start != info['records']: raise ValueError('Corpus packaging mismatch')
    Path(target_manifest).write_text(json.dumps(dict(corpus=info,pages=pages),indent=2),encoding='utf-8')
    print(f'Packaged {start:,} canonical records into {len(pages)} independently verified pages.')

if __name__ == '__main__':
    package('.corpus-build', '.corpus-pages', 'trainer/corpus-manifest.json')
