import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import venv
from contextlib import contextmanager

TRAINER_VERSION = 'full-grad-v2'
ROOT = Path(__file__).resolve().parents[1]
# Directly executing a compiled trainer/start.pyc does not put the payload root
# on sys.path. The Windows EXE bundles trainer/*.pyc without .py sources.
sys.path.insert(0, str(ROOT))
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


from trainer.progress import SetupProgress

def run_setup_step(label, command, meter, estimate, logfile):
    """Keep console readable while persisting the full diagnostic log."""
    started = time.monotonic()
    with open(logfile, 'ab') as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                while True:
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        meter.update(label, time.monotonic()-started, estimate)
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.terminate()
                raise
    if code:
        lines=Path(logfile).read_text(errors='replace').splitlines()[-18:]
        raise RuntimeError(f'{label} failed ({code}). Details in {logfile}:\n'+'\n'.join(lines))
    meter.finish_stage(label, estimate)


def prepare_environment(gpu, repair, teacher=False):
    device = 'gpu' if gpu else 'cpu'
    app_data = os.environ.get('LOCALAPPDATA')
    if os.name == 'nt' and Path(__file__).suffix == '.pyc' and app_data:
        folder = Path(app_data) / 'AXONAI' / 'environments' / f'python312-{device}'
    else:
        folder = ROOT / f'.trainer-{device}'
    python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    stamp = folder / 'dependencies.sha256'
    optional_stamp=folder / 'teacher.sha256'
    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes() + f'{TRAINER_VERSION}-{device}'.encode()).hexdigest()
    teacher_fingerprint = hashlib.sha256((ROOT / 'trainer' / 'requirements-teacher.txt').read_bytes() + b'teacher-v1').hexdigest() if teacher else None
    with setup_lock(folder):
        healthy = not repair and python.is_file() and stamp.is_file() and stamp.read_text().strip()==fingerprint
        teacher_ready = (not teacher or (optional_stamp.is_file() and optional_stamp.read_text().strip()==teacher_fingerprint))
        if healthy and teacher_ready:
            print('Dependencies already installed. Reusing AXONAI environment.', flush=True)
            return python
        # Cached wheels remain available in pip's normal shared cache.
        steps=[]
        if not python.is_file(): steps.append(('Python environment', 8))
        if not healthy:steps.append(('Trainer packages', 95))
        # The CUDA PyTorch runtime replaces the separate CuPy package.
        if not healthy: steps.append(('Full-gradient AI runtime', 160))
        if teacher and (repair or not teacher_ready):
            steps.append(('Local pretrained reviewer', 65))
        steps += [('Package verification', 8),('Runtime verification', 8)]
        meter=SetupProgress(steps)
        logfile=folder/'setup.log'
        if not python.is_file():
            meter.update('Python environment')
            venv.EnvBuilder(with_pip=True).create(folder)
            meter.finish_stage('Python environment', 8)
        pip = [str(python), '-u', '-m', 'pip', 'install', '--no-compile',
               '--no-input', '--disable-pip-version-check',
               '--prefer-binary', '--progress-bar', 'off',
               '--timeout', '30', '--retries', '3']
        if repair:
            pip.append('--force-reinstall')
        if not healthy:
            stamp.unlink(missing_ok=True)
            run_setup_step('Trainer packages', [*pip, '-r', str(REQUIREMENTS)],meter,95,logfile)
            index = 'https://download.pytorch.org/whl/cu128' if gpu else 'https://download.pytorch.org/whl/cpu'
            run_setup_step('Full-gradient AI runtime', [*pip, '--index-url', index, 'torch==2.7.1'],meter,160,logfile)
            stamp.write_text(fingerprint + '\n')
        if teacher and (repair or not teacher_ready):
            optional_stamp.unlink(missing_ok=True)
            run_setup_step('Local pretrained reviewer',[*pip,'-r',
                str(ROOT/'trainer'/'requirements-teacher.txt')],meter,65,logfile)
            optional_stamp.write_text(teacher_fingerprint + '\n')
        run_setup_step('Package verification',[str(python),'-m','pip','check'],meter,8,logfile)
        check='import numpy, psutil, requests, torch; from safetensors.torch import load_file, save_file'
        if teacher:check+='; import transformers'
        if gpu:check+='; assert torch.cuda.is_available(), "CUDA GPU unavailable"'

        run_setup_step('Runtime verification',[str(python),'-c',check],meter,8,logfile)
        meter.finish()
        print('Dependency setup complete.', flush=True)
    return python


def preflight(server):
    """Check the training API before installing multi-gigabyte dependencies."""
    import json
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    parsed = urlparse(server)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError('Use an HTTPS training server URL without credentials.')
    try:
        with urlopen(Request(server.rstrip('/') + '/api/status',
                             headers={'User-Agent': 'AXONAI-Trainer/2.0'}), timeout=15) as response:
            info = json.load(response)
    except (URLError, OSError, ValueError, TimeoutError) as error:
        raise RuntimeError('Cannot reach the training API. Check your internet connection '
                           'and the --server URL, then reopen the trainer. '
                           '(' + type(error).__name__ + ')') from error
    if info.get('ok') is not True or info.get('training_available') is not True:
        reason = str(info.get('training_status_reason') or 'storage or coordinator unavailable')[:100]
        raise RuntimeError('Training API is not ready (' + reason + '). '
                           'The website owner must fix its storage/connection first.')
    print('Training API preflight passed.', flush=True)


def main():
    args = sys.argv[1:]
    if '--help' in args:
        print('AXONAI console trainer: --gpu, --rounds N, --server HTTPS_URL, --yes, --install-only, --repair, --teacher, --no-teacher')
        return 0
    gpu = '--gpu' in args
    try:
        if '--install-only' not in args and '--repair' not in args:
            server = args[args.index('--server')+1] if '--server' in args and args.index('--server')+1 < len(args) else 'https://axonai-theta.vercel.app'
            preflight(server)
        wants_teacher = '--teacher' in args and '--no-teacher' not in args
        try:
            python = prepare_environment(gpu, '--repair' in args, teacher=wants_teacher)
        except RuntimeError as error:
            if not wants_teacher or not any(
                label in str(error) for label in
                ('Local pretrained reviewer failed', 'Runtime verification failed')
            ):
                raise
            print('Optional pretrained checker setup failed. Starting student training without it.', flush=True)
            wants_teacher = False
            python = prepare_environment(gpu, False, teacher=False)
        if '--install-only' in args:
            return 0
        entry = ROOT / 'trainer' / 'run.py'
        if not entry.exists():
            entry = entry.with_suffix('.pyc')
        options = [arg for arg in args if arg not in ('--gpu', '--repair', '--teacher-check')]
        if not wants_teacher and '--no-teacher' not in options:
            options.append('--no-teacher')
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
