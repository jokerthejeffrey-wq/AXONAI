import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from trainer.chunked_corpus import load_manifest, read_assigned
from trainer.corpus import CORPUS_ID, REVISION

class ChunkedCorpusTests(unittest.TestCase):
    def fixture(self, folder):
        pages=[]; content={}
        for start in (0,3):
            raw=b''.join((json.dumps({'id':i})+'\n').encode() for i in range(start,start+3))
            blob=gzip.compress(raw,mtime=0);name=f'page-{start:07d}.jsonl.gz'
            content[name]=blob
            pages.append(dict(start=start,count=3,name=name,size=len(blob),raw_size=len(raw),sha256=hashlib.sha256(blob).hexdigest()))
        data=dict(corpus=dict(id=CORPUS_ID,revision=REVISION,records=6),pages=pages)
        path=Path(folder)/'manifest.json';path.write_text(json.dumps(data))
        return load_manifest(path),content

    def test_only_assigned_pages_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,content=self.fixture(d);calls=[]
            def get(url,**kw):
                calls.append(url);return io.BytesIO(content[url.rsplit('/',1)[-1]])
            self.assertEqual(read_assigned(d,manifest,1,1,get),[{'id':1}])
            self.assertEqual(len(calls),1)
            self.assertEqual(read_assigned(d,manifest,2,3,get),[{'id':i} for i in (2,3,4)])
            self.assertEqual(len(calls),2)
            read_assigned(d,manifest,0,6,get)
            self.assertEqual(len(calls),2)

    def test_corrupt_download_never_enters_cache(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,_=self.fixture(d)
            with self.assertRaisesRegex(ValueError,'checksum'):
                read_assigned(d,manifest,0,1,lambda *a,**kw:io.BytesIO(b'broken'))
            self.assertEqual(list((Path(d)/'pages').iterdir()),[])

    def test_repair_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,content=self.fixture(d)
            get=lambda url,**kw:io.BytesIO(content[url.rsplit('/',1)[-1]])
            read_assigned(d,manifest,0,1,get)
            (Path(d)/'pages'/manifest['pages'][0]['name']).write_bytes(b'broken')
            self.assertEqual(read_assigned(d,manifest,0,1,get),[{'id':0}])

    def test_reject_bad_ranges_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,_=self.fixture(d)
            for start,count in [(-1,1),(5,2),(0,0)]:
                with self.assertRaises(ValueError):read_assigned(d,manifest,start,count)
            manifest['pages'][0]['name']='../outside'
            p=Path(d)/'bad.json';p.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_manifest(p)

if __name__=='__main__':unittest.main()
