"""Run with Python 3.12 on a Windows build runner."""
import hashlib,py_compile,subprocess,os,zipfile,shutil,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'exe-build';out.mkdir(exist_ok=True)
stage=out/'payload';stage.mkdir(exist_ok=True)
files=[root/'settings.py']+[p for p in (root/'trainer').glob('*.py') if not p.name.startswith('test_') and p.name != 'teacher.py']
for src in files:
 dst=stage/src.relative_to(root).with_suffix('.pyc');dst.parent.mkdir(parents=True,exist_ok=True)
 py_compile.compile(str(src),cfile=str(dst),dfile=str(src.relative_to(root)),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
shutil.copy2(root/'trainer/requirements.txt',stage/'trainer/requirements.txt')
shutil.copy2(root/'LICENSE',stage/'LICENSE')
payload=out/'payload.zip'
with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(stage.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(stage).as_posix())
installer=out/'python-installer.exe'
urllib.request.urlretrieve('https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe',installer)
# Verify the official installer signature before pinning its bytes into the launcher.
subprocess.run(['pwsh','-NoProfile','-Command',"$s=Get-AuthenticodeSignature '"+str(installer)+"'; if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'Python Software Foundation'){throw 'Invalid Python installer signature'}"],check=True)
installer_hash=hashlib.sha256(installer.read_bytes()).hexdigest();payload_hash=hashlib.sha256(payload.read_bytes()).hexdigest()
csc=Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
for device in ('CPU','GPU'):
 source=(root/'scripts/windows/Launcher.cs').read_text().replace('__DEVICE__',device).replace('__INSTALLER_HASH__',installer_hash).replace('__PAYLOAD_HASH__',payload_hash)
 cs=out/(device+'.cs');cs.write_text(source)
 subprocess.run([str(csc),'/nologo','/target:exe','/platform:x64','/optimize+','/r:System.IO.Compression.dll','/r:System.IO.Compression.FileSystem.dll','/resource:'+str(payload)+',payload.zip','/out:'+str(out/(device+'-AXON.exe')),str(cs)],check=True)
 subprocess.run([str(out/(device+'-AXON.exe')),'--self-test'],check=True)
# Verify a real interpreter install/detection with the same launcher path used by users.
subprocess.run([str(out/'CPU-AXON.exe'),'--setup-only'],check=True)
print('Both console executables built and verified.')
