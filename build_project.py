"""Owner packaging: run after intentional source/UI edits, before redistributing."""
from pathlib import Path
import hashlib,json,shutil,zipfile
root=Path(__file__).resolve().parent
names=['GPU.py','CPU.py','bootstrap.py','start.ps1','START_CPU.cmd','START_GPU.cmd','client.py','student.py','settings.py','requirements.txt','CLIENT_README.md']
(root/'client_manifest.json').write_text(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
target=root/'vercel/public';target.mkdir(parents=True,exist_ok=True)
for file in (root/'templates').glob('*.html'):shutil.copy2(file,target/file.name)
shutil.copytree(root/'static',target/'static',dirs_exist_ok=True)
(target/'config.js').write_text('window.AXON_API=null; // Set AXON_API_URL at Vercel build time.\n')
print('Client integrity manifest and static Vercel assets rebuilt.')

(target/'download').mkdir(exist_ok=True)
with zipfile.ZipFile(target/'download/client.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names+['client_manifest.json']:z.write(root/name,'AXONAI_Client/'+name)
print('Static client ZIP prepared for Vercel downloads.')
