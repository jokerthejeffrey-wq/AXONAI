import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import venv
from contextlib import contextmanager

TRAINER_VERSION = 'numpy-v1'
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / 'trainer' / 'requirements.txt'


@contextmanager
def setup_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'setup.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno != 13:
                        raise
                    print('Another AXONAI window is setting up this environment. Waiting…', flush=True)
                    time.sleep(15)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield  # Closing the file releases the lock, including after cancellation.


def run_setup_step(label, command):
    print(label, flush=True)
    started = time.monotonic()
    with subprocess.Popen(command) as process:
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    print(f'{label} Still running ({elapsed}s elapsed).', flush=True)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            raise
    if code:
        raise RuntimeError(f'{label} Failed with exit code {code}. See the output above.')


def prepare_environment(gpu, repair):
    device = 'gpu' if gpu else 'cpu'
    app_data = os.environ.get('LOCALAPPDATA')
    if os.name == 'nt' and Path(__file__).suffix == '.pyc' and app_data:
        folder = Path(app_data) / 'AXONAI' / 'environments' / f'python312-{device}'
    else:
        folder = ROOT / f'.trainer-{device}'
    python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    stamp = folder / 'dependencies.sha256'
    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes() + f'{TRAINER_VERSION}-{device}'.encode()).hexdigest()
    with setup_lock(folder):
        if not repair and python.is_file() and stamp.is_file() and stamp.read_text().strip() == fingerprint:
            print('Dependencies already installed. Reusing AXONAI environment.', flush=True)
            return python
        stamp.unlink(missing_ok=True)
        if not python.is_file():
            print('Creating the AXONAI Python environment…', flush=True)
            venv.EnvBuilder(with_pip=True).create(folder)
        pip = [str(python), '-u', '-m', 'pip', 'install', '--no-compile', '--no-input',
               '--disable-pip-version-check', '--timeout', '30', '--retries', '3']
        if repair:
            pip.append('--force-reinstall')
        run_setup_step('Installing trainer dependencies.', [*pip, '-r', str(REQUIREMENTS)])
        if gpu:
            run_setup_step('Installing NVIDIA GPU array support.', [*pip, 'cupy-cuda12x==13.5.1'])
        run_setup_step('Checking package compatibility.', [str(python), '-m', 'pip', 'check'])
        check = 'import numpy, pyarrow, psutil, requests, safetensors'
        if gpu:
            check += '; import cupy; assert cupy.cuda.runtime.getDeviceCount() > 0'
        run_setup_step('Checking the training runtime.', [str(python), '-c', check])
        stamp.write_text(fingerprint + '\n')
        print('Dependency setup complete.', flush=True)
    return python


def main():
    args = sys.argv[1:]
    if '--help' in args:
        print('AXONAI console trainer: --gpu, --rounds N, --server HTTPS_URL, --yes, --install-only, --repair')
        return 0
    gpu = '--gpu' in args
    try:
        python = prepare_environment(gpu, '--repair' in args)
        if '--install-only' in args:
            return 0
        entry = ROOT / 'trainer' / 'run.py'
        if not entry.exists():
            entry = entry.with_suffix('.pyc')
        options = [arg for arg in args if arg not in ('--gpu', '--repair')]
        return subprocess.run([str(python), '-u', str(entry), '--device', 'cuda' if gpu else 'cpu', *options]).returncode
    except KeyboardInterrupt:
        print('\nSetup stopped. Open the trainer again to resume with the cached downloads.', flush=True)
        return 130
    except (OSError, RuntimeError) as error:
        print(f'\nAXONAI setup: {error}', file=sys.stderr, flush=True)
        print('You can retry by reopening the EXE. For damaged packages, run the EXE with --repair.', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
