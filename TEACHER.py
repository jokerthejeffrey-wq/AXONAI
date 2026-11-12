"""Owner-only synthetic data authoring. NEVER imported by CHAT.py or student.py."""
import sys
if __name__=='__main__':
    from bootstrap import ensure
    ensure('teacher-gpu' if '--gpu' in sys.argv else 'teacher','TEACHER.py')
import json,argparse,hashlib
from settings import ROOT,TEACHER,TEACHER_REVISION,CONFIG

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--limit',type=int,default=1000)
    args=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if args.gpu and not torch.cuda.is_available():raise SystemExit('CUDA unavailable: update NVIDIA driver or rerun without --gpu.')
    device='cuda' if args.gpu else 'cpu';torch.set_num_threads(2)
    print('Training-only teacher: downloading/loading',TEACHER,flush=True)
    tok=AutoTokenizer.from_pretrained(TEACHER,revision=TEACHER_REVISION,trust_remote_code=False)
    teacher=AutoModelForCausalLM.from_pretrained(TEACHER,revision=TEACHER_REVISION,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32).to(device).eval()
    curriculum=json.loads((ROOT/'data/curriculum.json').read_text(encoding='utf-8'))
    destination=ROOT/'data/teacher_drafts.jsonl';seen=set()
    if destination.exists():
        for line in destination.read_text(encoding='utf-8').splitlines():
            if line.strip():seen.add(json.loads(line)['source_id'])
    count=0
    for row in curriculum:
        identity=hashlib.sha256(json.dumps(row['messages'],sort_keys=True).encode()).hexdigest()
        if identity in seen:continue
        if count>=args.limit:break
        messages=[{'role':'system','content':'You write training replies for a separate student assistant named AXONAI, not for Qwen. AXONAI is developed by a community of developers, testers, and compute contributors; never claim Anthropic or OpenAI created AXONAI. You are a helpful tutor producing a short, accurate assistant reply for a small student model. Answer naturally in one or two short sentences, under 35 words. Continue the conversation, answer follow-ups, and respect requests to end. Never add labels or commentary about training.'}]+row['messages']
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
        with torch.inference_mode():
            out=teacher.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=True,temperature=.8,top_p=.9,pad_token_id=tok.eos_token_id)
        answer=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
        prefix=sum(2+len(m['content'].encode('utf-8')) for m in row['messages'])+1
        available=CONFIG['context']-prefix
        if available<32:print('Skipped overlong context:',row['concept']);continue
        answer=answer.encode('utf-8')[:available].decode('utf-8',errors='ignore').strip()
        if not answer:continue
        record=dict(concept=row['concept'],messages=row['messages'],response=answer,source_id=identity,teacher=TEACHER,teacher_revision=TEACHER_REVISION)
        with destination.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        seen.add(identity);count+=1
        print(f"[{count}] {row['concept']}: {answer}",flush=True)
    print('Drafts saved, NOT imported into training. Run: python REVIEW.py, then python manage.py init (or import).')
    print('Pretrained weights are not copied into the student. Website chat may use separately labeled pretrained assistance.')

if __name__=='__main__':main()
