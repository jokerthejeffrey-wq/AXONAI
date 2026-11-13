import os,sys,subprocess,venv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if '--help' in sys.argv:
    print('AXONAI console trainer: --gpu, --rounds N, --server HTTPS_URL, --yes');raise SystemExit(0)
gpu='--gpu' in sys.argv
folder=root/('.trainer-gpu' if gpu else '.trainer-cpu')
python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if not python.exists():venv.EnvBuilder(with_pip=True).create(folder)
subprocess.run([str(python),'-m','pip','install','torch==2.7.1','--index-url','https://download.pytorch.org/whl/'+('cu128' if gpu else 'cpu')],check=True)
subprocess.run([str(python),'-m','pip','install','-r',str(root/'trainer/requirements.txt')],check=True)
subprocess.run([str(python),str(root/('trainer/run.py' if (root/'trainer/run.py').exists() else 'trainer/run.pyc')),'--device','cuda' if gpu else 'cpu',*[a for a in sys.argv[1:] if a!='--gpu']],check=True)
