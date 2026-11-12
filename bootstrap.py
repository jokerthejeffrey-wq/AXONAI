"""Console setup. Repairs an isolated environment, never rewrites user/system Python."""
import os,sys,subprocess,venv,json,hashlib
from urllib.parse import urlparse
from urllib.request import urlopen
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def repair_sources():
    """Restore only files whose hashes match the trusted manifest shipped in this ZIP."""
    manifest=ROOT/'client_manifest.json'
    if not manifest.exists():
        raise SystemExit('Missing client_manifest.json. Re-extract the complete trusted ZIP before running.')
    expected=json.loads(manifest.read_text())
    allowed={'GPU.py','CPU.py','bootstrap.py','start.ps1','START_CPU.cmd','START_GPU.cmd','client.py','student.py','settings.py','requirements.txt','CLIENT_README.md'}
    if set(expected)!=allowed:raise SystemExit('Invalid manifest. Re-download the trusted complete ZIP.')
    damaged=[name for name,digest in expected.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if not damaged:return
    print('Missing or changed client files:',', '.join(damaged))
    print('Restoring will overwrite local edits to these files with the version shipped in this ZIP.')
    if input('Restore matching files from your project backend? [y/N] ').lower()!='y':raise SystemExit('Cancelled; re-extract the ZIP or restore your source files.')
    base=os.getenv('AXON_SERVER','')
    if '--server' in sys.argv:
        i=sys.argv.index('--server')
        if i+1<len(sys.argv):base=sys.argv[i+1]
    base=(base or input('Trusted backend HTTPS URL: ').strip()).rstrip('/')
    url=urlparse(base)
    if url.scheme!='https' and not(url.scheme=='http' and url.hostname in ('127.0.0.1','localhost')):raise SystemExit('Remote repairs require HTTPS.')
    for name in damaged:
        with urlopen(base+'/download/source/'+name,timeout=60) as response:data=response.read(1024*1024+1)
        if len(data)>1024*1024 or hashlib.sha256(data).hexdigest()!=expected[name]:
            raise SystemExit('Repair hash mismatch. No replacement made for '+name+'. Download a fresh complete ZIP if the server version changed.')
        temp=ROOT/(name+'.repair');temp.write_bytes(data);temp.replace(ROOT/name)
    print('Source files restored. Restart this launcher to load the verified files.')
    raise SystemExit(0)

def ensure(kind,entry):
    if os.getenv('AXON_BOOTSTRAPPED')=='1':return
    repair_sources()
    if sys.version_info[:2] not in ((3,11),(3,12)):
        raise SystemExit('Use Python 3.11/3.12. On Windows run START_CPU.cmd or START_GPU.cmd to install Python 3.12.')
    env=ROOT/('.axon-'+kind)
    python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    requirements=(ROOT/'requirements.txt').read_bytes()
    if kind.startswith('teacher'):requirements+=(ROOT/'requirements-teacher.txt').read_bytes()
    signature='v2-'+kind+'-'+hashlib.sha256(requirements).hexdigest()
    stamp=env/'axon-ready.json'
    check='import torch,requests,numpy,safetensors,flask,waitress;'+('import transformers;' if kind.startswith('teacher') else '')
    healthy=False
    if python.exists() and stamp.exists():
        try:healthy=json.loads(stamp.read_text()).get('signature')==signature and subprocess.run([str(python),'-c',check],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        except (ValueError,OSError):pass
    if not healthy:
        print('\nAXONAI setup: installs/repairs packages in',env)
        print('Downloads can be several GB. No system Python changes. GPU drivers are not installed.')
        if input('Continue installation? [y/N] ').strip().lower()!='y':raise SystemExit('Cancelled')
        runtime_ok=False
        if python.exists():
            try:runtime_ok=subprocess.run([str(python),'-c','import sys; print(sys.version)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15).returncode==0
            except (OSError,subprocess.TimeoutExpired):pass
        if not runtime_ok:
            print('Creating/rebuilding the isolated Python environment…')
            venv.EnvBuilder(with_pip=True,clear=env.exists()).create(env)
        try:
            run([python,'-m','ensurepip','--upgrade'])
            run([python,'-m','pip','install','--upgrade','pip'])
            index='https://download.pytorch.org/whl/cu128' if kind.endswith('gpu') else 'https://download.pytorch.org/whl/cpu'
            index_args=[] if sys.platform=='darwin' else ['--index-url',index]
            for attempt in range(2):
                try:
                    repair=['--force-reinstall'] if attempt else []
                    run([python,'-m','pip','install',*repair,'torch==2.7.1',*index_args])
                    req='requirements-teacher.txt' if kind.startswith('teacher') else 'requirements.txt'
                    run([python,'-m','pip','install',*repair,'-r',ROOT/req])
                    run([python,'-c',check]);break
                except subprocess.CalledProcessError:
                    if attempt:raise
                    print('Retrying once with clean package reinstall…')
            stamp.write_text(json.dumps({'signature':signature}))
        except (OSError,subprocess.CalledProcessError) as e:
            raise SystemExit(f'Setup failed: {e}\nCheck internet, free disk space, Python architecture and antivirus logs. Re-run to retry.')
    envvars=os.environ.copy();envvars['AXON_BOOTSTRAPPED']='1'
    result=subprocess.run([str(python),str(ROOT/entry),*sys.argv[1:]],env=envvars,cwd=ROOT)
    raise SystemExit(result.returncode)
