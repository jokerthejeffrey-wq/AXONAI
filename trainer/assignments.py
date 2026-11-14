"""Continuous volunteer job scheduling: zero session or daily job caps.

A finite source corpus is never replayed. Queue congestion causes a pause
rather than ending the session. Ctrl+C interrupts the pause immediately.
"""
import time
from itertools import count

def rounds(limit=0):
    if not isinstance(limit,int) or limit<0:
        raise ValueError('Rounds must be 0 (unlimited) or a positive number')
    return count(0) if limit==0 else range(limit)

def claim_until_ready(get_job, sleep=time.sleep, printer=print):
    while True:
        response=get_job()
        if response.status_code!=204:
            return response
        reason=response.headers.get('X-AXON-Queue','review-pending')
        if reason=='corpus-complete':
            printer('All unique training records have been assigned. Source corpus exhausted.')
            return None
        try:
            seconds=int(response.headers.get('Retry-After','45'))
        except (TypeError,ValueError):
            seconds=45
        seconds=min(300,max(15,seconds))
        printer('Training temporarily paused ('+reason+'). Checking again in '+str(seconds)+'s. Ctrl+C to stop.',flush=True)
        sleep(seconds)
