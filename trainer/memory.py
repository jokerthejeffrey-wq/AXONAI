import time
from pathlib import Path
import psutil

def wait(reserve_mb=256):
    while True:
        v=psutil.virtual_memory();checks=[(v.total,v.total-v.available)]
        try:
            cap=int(Path('/sys/fs/cgroup/memory.max').read_text());used=int(Path('/sys/fs/cgroup/memory.current').read_text());checks.append((cap,used))
        except (OSError,ValueError):pass
        if all(used+reserve_mb*1024**2<cap*.90 for cap,used in checks):return
        print('RAM above admission budget; waiting 5 seconds. Ctrl+C stops.',flush=True);time.sleep(5)
