"""Owner commands. Stop HTML.py before init/import/merge/reject/rollback."""
if __name__=='__main__':
    from bootstrap import ensure
    ensure('cpu','manage.py')
import argparse,json,re
from settings import ROOT,STATE
from ledger import connect,version,checkpoint,import_lessons

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','import','list','budget','merge','reject','rollback']);p.add_argument('ids',nargs='*');a=p.parse_args()
    if a.command in ('init','import'):
        if not (ROOT/'data/train.jsonl').exists():raise SystemExit('Run TEACHER.py first, then review data/train.jsonl.')
        print('Lesson records read:',import_lessons())
        if a.command=='import':return
    if a.command=='list':
        with connect() as c:
            for r in c.execute('SELECT * FROM updates ORDER BY created DESC LIMIT 100'):print(dict(r))
        print('Version:',version());return
    if a.command=='budget':
        with connect() as c:
            for r in c.execute('SELECT * FROM concepts ORDER BY name'):print(r['name'],f"{r['uses']}/4 reserved uses")
        return
    if a.command=='rollback':
        if len(a.ids)!=1 or not a.ids[0].isdigit() or not checkpoint(int(a.ids[0])).exists():p.error('Supply an existing version.')
        v=version()+1
        while checkpoint(v).exists():v+=1
        checkpoint(v).write_bytes(checkpoint(int(a.ids[0])).read_bytes())
        with connect() as c:c.execute("UPDATE meta SET value=? WHERE key='version'",(str(v),))
        print('Rolled back as NEW version',v,'; training budgets remain consumed.');return
    if a.command in ('merge','reject'):
        if not a.ids or len(set(a.ids))!=len(a.ids) or any(not re.fullmatch('[0-9a-f]{32}',x) for x in a.ids):p.error('Supply unique update IDs from list.')
        with connect() as c:
            for uid in a.ids:
                r=c.execute('SELECT * FROM updates WHERE id=?',(uid,)).fetchone()
                if not r or r['status']!='pending' or (a.command=='merge' and r['version']!=version()):p.error('Unknown, reviewed or stale update.')
            if a.command=='reject':
                for uid in a.ids:
                    c.execute("UPDATE updates SET status='rejected' WHERE id=?",(uid,));(STATE/'pending'/f'{uid}.safetensors').unlink(missing_ok=True)
                return
    import torch
    from student import Student,snapshot,load_weights,evaluate
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(2);torch.manual_seed(2026);model=Student()
    if a.command=='init':
        if checkpoint(version()).exists():print('Already initialized; budgets preserved.');return
        if version()!=0:raise RuntimeError('Missing checkpoint: restore a backup.')
        checkpoint(0).parent.mkdir(parents=True,exist_ok=True);save_file(snapshot(model),str(checkpoint(0)))
        print('Random student initialized:',sum(p.numel() for p in model.parameters()),'parameters. No pretrained weights used.');return
    v=version();load_weights(model,checkpoint(v));old=snapshot(model)
    rows=[json.loads(s) for s in (ROOT/'data/validation.jsonl').read_text().splitlines() if s.strip()]
    before=evaluate(model,rows);average={k:torch.zeros_like(t) for k,t in old.items()}
    for uid in a.ids:
        path=STATE/'pending'/f'{uid}.safetensors';load_weights(model,path);incoming=snapshot(model)
        delta=sum(float((incoming[k]-old[k]).double().square().sum()) for k in old)
        if delta/sum(t.numel() for t in old.values())>.05**2:raise ValueError('Excessive update magnitude')
        for k in old:average[k]+=incoming[k]/len(a.ids)
    candidate={k:old[k]*.5+average[k]*.5 for k in old};model.load_state_dict(candidate)
    after=evaluate(model,rows);print(f'Holdout loss {before:.5f} -> {after:.5f}')
    if not(after==after and after<=before):print('Not published. No holdout improvement; uses are NOT reset.');return
    new=v+1
    while checkpoint(new).exists():new+=1
    temp=checkpoint(new).with_suffix('.tmp');save_file(candidate,str(temp));temp.replace(checkpoint(new))
    with connect() as c:
        c.execute("UPDATE meta SET value=? WHERE key='version'",(str(new),))
        for uid in a.ids:c.execute("UPDATE updates SET status='accepted' WHERE id=?",(uid,))
    for uid in a.ids:(STATE/'pending'/f'{uid}.safetensors').unlink(missing_ok=True)
    print('Published student version',new,'. Restart HTML.py. Early answers may still be nonsense.')
if __name__=='__main__':main()
