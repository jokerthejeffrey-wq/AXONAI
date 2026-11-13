"""Terminal progress indicators for bounded setup tasks; ETA is explicitly approximate."""
import sys
import time


def duration(seconds):
    if seconds is None:
        return 'calculating'
    seconds = max(0, int(round(seconds)))
    if seconds >= 3600:
        return f'~{seconds // 3600}h {(seconds % 3600) // 60}m'
    if seconds >= 60:
        return f'~{seconds // 60}m {seconds % 60}s'
    return f'~{seconds}s'


def bar(percent, label='', eta=None, width=20):
    percent = min(100, max(0, int(percent)))
    filled = min(width, round(width * percent / 100))
    return f'{label} [{"=" * filled}{" " * (width - filled)}] {percent}%  remaining: {duration(eta)}'


class SetupProgress:
    """Weights describe finished setup stages, not invented pip download percentages.

    During an unknown-length subprocess, its stage approaches at most 90%.
    ETA comes from observed elapsed time and conservative per-stage estimates.
    An unseen download speed is never presented as a measured time.
    """
    def __init__(self, steps, stream=None, clock=None):
        self.steps = [(str(name), max(0.01, float(estimate))) for name, estimate in steps]
        self.total = sum(seconds for _, seconds in self.steps)
        self.done = 0.0
        self.started = (clock or time.monotonic)()
        self.clock = clock or time.monotonic
        self.stream = stream or sys.stdout
        self.last = 0
        self.width = 0

    def update(self, label, elapsed=0, estimate=1, complete=False):
        amount = float(estimate) if complete else min(.90, max(0, elapsed / max(estimate * 2, elapsed * 1.1)) * estimate)
        fraction = (self.done + amount) / self.total
        percent = 100 if complete and self.done + estimate >= self.total - .01 else int(fraction * 100)
        eta = None
        if self.clock() - self.started > 3 and fraction > .03:
            remaining = max(0, (1 - fraction) * (self.clock() - self.started) / max(.01, fraction))
            eta = min(24 * 3600, remaining)
        line = bar(percent, label, eta)
        self.stream.write('\r' + line.ljust(self.width))
        self.stream.flush()
        self.width = max(self.width, len(line))

    def finish_stage(self, label, estimate):
        self.done = min(self.total, self.done + estimate)
        self.update(label, 0, 0.001, complete=False)
        self.stream.write('\n')
        self.stream.flush()
        self.width = 0

    def finish(self):
        self.done = self.total
        self.update('Setup complete', 0, 0.001, complete=True)
        self.stream.write('\n')
        self.stream.flush()
