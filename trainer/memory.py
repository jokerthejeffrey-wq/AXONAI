"""RAM admission checks for local AXON training.

Admission is deliberately conservative: do not start a new allocation if
physical or cgroup RAM is likely to exceed 90%. A constrained machine gets
a bounded wait and a useful explanation instead of an endless console loop.
"""
import time
from pathlib import Path

import psutil

LIMIT_FRACTION = 0.90
MB = 1024 * 1024


class MemoryAdmissionError(RuntimeError):
    pass


def budgets(reserve_mb=256):
    if not isinstance(reserve_mb, (int, float)) or reserve_mb < 0:
        raise ValueError('Invalid RAM reservation')
    reserve = int(reserve_mb * MB)
    v = psutil.virtual_memory()
    checks = [('system', int(v.total), max(0, int(v.total - v.available)))]
    # Container limits count only if they are smaller than the host's RAM.
    # This check is harmless on Windows, where /sys/fs/cgroup is absent.
    try:
        raw = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if raw != 'max':
            cap = int(raw)
            used = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
            if 0 < cap < v.total:
                checks.append(('container', cap, max(0, used)))
    except (OSError, ValueError):
        pass
    result = []
    for name, cap, used in checks:
        safe = max(0, int(cap * LIMIT_FRACTION) - used)
        result.append({'name': name, 'total': cap, 'used': used,
                       'safe_available': safe,
                       'reserve': reserve,
                       'allowed': reserve <= safe})
    return result


def wait(reserve_mb=256, max_wait_seconds=75, interval=5, *,
         clock=None, sleep=None, output=None):
    """Wait briefly for enough RAM, or exit with a measured explanation.

    reserve_mb estimates *new* memory required by the next operation; it is
    not the overall maximum memory limit for the process.
    """
    if max_wait_seconds < 0 or interval <= 0:
        raise ValueError('Invalid memory wait interval')
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    import sys
    output = output or sys.stdout
    started = clock()
    previous_length = 0
    displayed = False
    while True:
        state = budgets(reserve_mb)
        failing = next((item for item in state if not item['allowed']), None)
        if failing is None:
            if displayed:
                output.write('\r' + 'RAM available; continuing.'.ljust(previous_length) + '\n')
                output.flush()
            return
        elapsed = max(0.0, clock() - started)
        free = failing['safe_available'] / MB
        need = failing['reserve'] / MB
        used_percent = 100.0 * failing['used'] / failing['total']
        if elapsed >= max_wait_seconds:
            if displayed:
                output.write('\n')
            raise MemoryAdmissionError(
                f'Not enough RAM to keep usage below 90%: '
                f'{used_percent:.0f}% already used, '
                f'{free:.0f} MB safe headroom, {need:.0f} MB needed. '
                'Training stopped instead of waiting indefinitely. '
                'Close other memory-heavy applications and start the trainer again.')
        remaining = max(0, int(max_wait_seconds - elapsed))
        line = (f'RAM {used_percent:.0f}% | safe free {free:.0f} MB / '
                f'need {need:.0f} MB | retrying ({remaining}s left)')
        output.write('\r' + line.ljust(previous_length))
        output.flush()
        previous_length = max(previous_length, len(line))
        displayed = True
        sleep(min(interval, max_wait_seconds - elapsed))
