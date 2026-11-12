import {spawnSync} from 'node:child_process';
const out=spawnSync(process.platform==='win32'?'npm.cmd':'npm',['ci','--no-audit','--no-fund'],{stdio:'inherit',shell:process.platform==='win32'});
process.exit(out.status??1);
