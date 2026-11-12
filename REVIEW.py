"""Review teacher drafts before they can enter training. Standard library only."""
import json
from settings import ROOT,CONFIG

def main():
    draft=ROOT/'data/teacher_drafts.jsonl';train=ROOT/'data/train.jsonl';rejected=ROOT/'data/rejected_drafts.json'
    if not draft.exists():raise SystemExit('Run TEACHER.py first.')
    seen={json.loads(s)['source_id'] for s in train.read_text(encoding='utf-8').splitlines() if s.strip()} if train.exists() else set()
    rejects=set(json.loads(rejected.read_text())) if rejected.exists() else set()
    total=0
    for line in draft.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);identity=row['source_id']
        if identity in seen or identity in rejects:continue
        print('\nTOPIC:',row['concept'])
        for m in row['messages']:print(m['role'].upper()+':',m['content'])
        print('TEACHER DRAFT:',row['response'])
        choice=input('[y] approve / [e] edit / [r] reject / Enter skip / [q] quit: ').strip().lower()
        if choice=='q':break
        if choice=='r':
            rejects.add(identity);rejected.write_text(json.dumps(sorted(rejects)));continue
        if choice=='e':row['response']=input('Correct response (single line): ').strip()
        elif choice!='y':continue
        size=sum(2+len(m['content'].encode('utf-8')) for m in row['messages'])+len(row['response'].encode('utf-8'))+2
        if not row['response'] or size>CONFIG['context']+1:print('Empty or overlong response: skipped.');continue
        row['review']='Approved in console review'
        with train.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        seen.add(identity);total+=1
    print('Approved',total,'new lessons. Stop the server, then run manage.py init or manage.py import.')
if __name__=='__main__':main()
