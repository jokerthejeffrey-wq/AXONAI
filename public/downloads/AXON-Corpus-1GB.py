#!/usr/bin/env python3
"""Generate AXON AI's reproducible ~1 GiB TSV training corpus locally.

This is PROCEDURAL data, not a pretrained language model or a claim of
15 million human-authored conversations. AXON CPU/GPU trainers get individual
rows from the website and do not require this whole file.
"""
import argparse
import shutil
from pathlib import Path

TARGET_BYTES = 1_073_741_824
MIN_LINES = 15_000_001

def lesson(n):
    if n == 0:
        return "hello", "Hello!"
    op = (n-1) % 24
    k = (n-1)//24 + 2
    b = (k*7)%113 + 3
    c = (k*11)%97 + 2
    small = k%50 + 2
    cases = [
        (f"Hi! My event starts in {k} minutes. Say hello.", "Hello! I hope your event goes well."),
        (f"What is {k} plus {b}?", str(k+b)),
        (f"Calculate {k+b} minus {b}.", str(k)),
        (f"What is {k} times {small}?", str(k*small)),
        (f"Divide {k*small} by {small}.", str(k)),
        (f"Is {k} even or odd?", "odd" if k%2 else "even"),
        (f"Which is greater, {k} or {k+b}?", str(k+b)),
        (f"What is {k} squared?", str(k*k)),
        (f"Write {k} as a decimal integer.", str(k)),
        (f"How many centimeters in {k} meters?", str(k*100)),
        (f"How many minutes are {k} hours?", str(k*60)),
        (f"How many grams are {k} kilograms?", str(k*1000)),
        (f"What is {k*small+b} mod {small}?", str(b%small)),
        (f"What is twice {k}?", str(k*2)),
        (f"What is three times {k}?", str(k*3)),
        (f"Continue: {k}, {k+b}, {k+2*b}, ...", str(k+3*b)),
        (f"What is the difference between {k+b} and {k}?", str(b)),
        (f"Evaluate {k} + {b} * 2.", str(k+b*2)),
        (f"Is {k} smaller than {k+c}?", "Yes."),
        (f"Write decimal {k} in binary.", format(k,"b")),
        (f"Round {k*10+b} to the nearest ten.", str(((k*10+b+5)//10)*10)),
        (f"How many digits are in the number {k}?", str(len(str(k)))),
        (f"What is 10% of {k*10}?", str(k)),
        (f"If there are {k} apples and you add {b}, how many?", str(k+b)),
    ]
    return cases[op]

def main():
    p=argparse.ArgumentParser(description="Generate AXON AI Q/A text (prompt TAB answer).")
    p.add_argument("--output",default="AXON-Training-Data.txt")
    p.add_argument("--sample",type=int,default=0,help="Create N preview lines instead of the full dataset (1..100000)")
    args=p.parse_args()
    if args.sample and not (1<=args.sample<=100_000):
        p.error("--sample must be between 1 and 100000")
    path=Path(args.output).expanduser().resolve()
    if path.exists():p.error("File already exists: "+str(path))
    if not path.parent.is_dir():p.error("Output folder does not exist")
    if not args.sample and shutil.disk_usage(path.parent).free < TARGET_BYTES + 200_000_000:
        p.error("Not enough free disk space; need at least 1.2 GB")
    lines,bytes_written=0,0
    min_lines=args.sample or MIN_LINES
    min_bytes=0 if args.sample else TARGET_BYTES
    with path.open("x",encoding="utf-8",newline="\n",buffering=1024*1024) as output:
        while lines < min_lines or bytes_written < min_bytes:
            prompt,answer=lesson(lines)
            row=prompt+"\t"+answer+"\n"
            output.write(row)
            bytes_written+=len(row.encode("utf-8"))
            lines+=1
            if lines%1_000_000==0:
                print(f"Examples: {lines:,} | Size: {bytes_written/2**30:.3f} GiB",flush=True)
    print(f"Created: {path}\nExamples: {lines:,}\nBytes: {bytes_written:,}")

if __name__=="__main__":
    main()
