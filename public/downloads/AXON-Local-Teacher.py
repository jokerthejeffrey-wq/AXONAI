#!/usr/bin/env python3
"""Optional owner-side offline teacher for AXON AI.

Run an already installed local llama.cpp-compatible server on 127.0.0.1:8080.
No paid AI API is used. Answer examples are saved locally for inspection.
--upload sends the selected examples to YOUR website, authenticated by the
AXON_OWNER_TOKEN environment variable; never share the owner token.
"""
import argparse, json, os, sys, urllib.request, urllib.parse
from pathlib import Path

PROMPTS=[
 "Say hello to someone starting a new conversation.",
 "Explain in two sentences why the sky looks blue.",
 "How should I begin learning programming?",
 "What does a computer processor do?",
 "Explain the difference between RAM and disk storage.",
 "What is a variable in programming?",
 "How does an internet browser load a website?",
 "What does machine learning mean?",
 "Explain to a beginner what a training example is.",
 "How do neural networks improve with practice?",
 "What does it mean when a model is overfitting?",
 "Why is it useful to test a model on unseen examples?",
 "Explain what a tokenizer does.",
 "What is the difference between a word and a token?",
 "How do I solve a math problem step by step?",
 "What is the purpose of a computer operating system?",
 "Explain how a battery powers a device.",
 "What makes an answer trustworthy?",
 "How can I check whether a website is safe?",
 "Why should I keep passwords private?",
 "Explain what open-source software is.",
 "Why is electricity use important for AI training?",
 "How can people collaborate on a project?",
 "How should I plan a small personal project?",
 "Give a short and friendly farewell.",
 "Explain the meaning of the word 'curious'.",
 "What do I do when I do not know the answer?",
 "How can I make a clear to-do list?",
 "What is the difference between a fact and an opinion?",
 "Explain why accurate arithmetic matters.",
 "What is a GPU used for?",
 "What is a CPU used for?",
 "Explain why computer fans spin.",
 "What is the difference between training and inference?",
 "How can I explain a complex topic in simple words?",
 "What is a computer file?",
 "How do I safely back up important documents?",
 "What does HTTP stand for?",
 "What is the difference between a website and a server?",
 "Explain how software updates work.",
 "How do scientists use experiments?",
 "Why might an AI response be incorrect?",
 "What does it mean to reason from evidence?",
 "Explain the difference between a question and an instruction.",
 "Why is a diverse dataset useful?",
 "Describe a good habit for learning a language.",
 "What is a percentage?",
 "What does 'estimate' mean in mathematics?",
 "Explain why subtracting a number from itself gives zero.",
 "Explain the meaning of 'community-built'."
]

def post_json(url,body,headers=None):
    req=urllib.request.Request(url,data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type":"application/json",**(headers or {})},method="POST")
    with urllib.request.urlopen(req,timeout=120) as resp:
        return json.load(resp)

def main():
    p=argparse.ArgumentParser(description="Prepare high-quality examples using an AI already running locally.")
    p.add_argument("--local-url",default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--site",default="https://axonai-theta.vercel.app")
    p.add_argument("--count",type=int,default=16)
    p.add_argument("--output",default="AXON-Teacher-Examples.tsv")
    p.add_argument("--upload",action="store_true",help="Submit examples after generating, requires AXON_OWNER_TOKEN")
    args=p.parse_args()
    teacher=urllib.parse.urlparse(args.local_url)
    if teacher.scheme!="http" or teacher.hostname not in {"localhost","127.0.0.1"}:
        p.error("The pretrained teacher must run locally on 127.0.0.1")
    site=urllib.parse.urlparse(args.site)
    if site.scheme!="https" or not site.hostname or site.username or site.password:
        p.error("The AXON website must use HTTPS")
    if not (1<=args.count<=len(PROMPTS)):p.error("count must be 1.."+str(len(PROMPTS)))
    file=Path(args.output)
    if file.exists():p.error("Output file already exists")
    teacher_rows=[]
    for prompt in PROMPTS[:args.count]:
        payload={
            "model":"local",
            "messages":[
                {"role":"system","content":"Give one short accurate, clear and friendly answer in plain English. Avoid uncertainty when the answer is known; admit limits if not known. Do not invent citations. Do not include any private data."},
                {"role":"user","content":prompt}
            ],
            "max_tokens":110,"temperature":0.25,"stream":False
        }
        try:
            data=post_json(args.local_url,payload)
            answer=data["choices"][0]["message"]["content"].strip()
            if not 5<=len(answer)<=220 or "\n" in answer:continue
            teacher_rows.append({"question":prompt,"answer":answer})
            print("Prepared",len(teacher_rows),"/",args.count)
        except Exception as e:
            print("Teacher request stopped:",type(e).__name__,file=sys.stderr)
            break
    if not teacher_rows:raise SystemExit("No usable teacher examples were generated")
    with file.open("x",encoding="utf-8") as out:
        for row in teacher_rows:out.write(row["question"]+"\t"+row["answer"]+"\n")
    print("Saved:",file,"(inspect the file before uploading)")
    if args.upload:
        token=os.getenv("AXON_OWNER_TOKEN","")
        if len(token)<32:raise SystemExit("Set AXON_OWNER_TOKEN privately before using --upload")
        for start in range(0,len(teacher_rows),16):
            batch=teacher_rows[start:start+16]
            result=post_json(args.site.rstrip("/")+"/api/teacher-import",
                {"rows":batch},{"Authorization":"Bearer "+token})
            print("Imported:",result.get("accepted",0),"Duplicates:",result.get("duplicates",0))

if __name__=="__main__":
    main()
