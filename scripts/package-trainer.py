from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
out=root/'public/downloads/AXON-Conversational-Trainer.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((root/'trainer').glob('*')):
  if p.is_file():z.write(p,'AXON-Trainer/trainer/'+p.name)
 for name in ['student.py','settings.py','LICENSE']:
  z.write(root/name,'AXON-Trainer/'+name)
print(out)
