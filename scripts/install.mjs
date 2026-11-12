import {spawnSync} from 'node:child_process';
// ONNX's CPU binaries ship in the npm package; Vercel has no CUDA GPU.
const result=spawnSync(process.platform==='win32'?'npm.cmd':'npm',['ci','--no-audit','--no-fund'],{
 stdio:'inherit',shell:process.platform==='win32',env:{...process.env,ONNXRUNTIME_NODE_INSTALL_CUDA:'skip'}
});
process.exit(result.status??1);
