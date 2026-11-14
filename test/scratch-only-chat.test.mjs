import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {parseChatBody} from '../server/security.js';
test('Chat route imports scratch model and never invokes pretrained or browser models',()=>{
  const code=fs.readFileSync('api/chat.js','utf8');
  assert.match(code,/scratchStream/);
  assert.match(code,/loadStudent/);
  assert.doesNotMatch(code,/beginGateway|assistedAnswer|Qwen|browser_model_required/);
  const ui=fs.readFileSync('public/static/chat.js','utf8');
  assert.match(ui,/experimental-preview/);
  assert.doesNotMatch(ui,/new Worker\\(|generateFromCloud|generateInBrowser/);
});
test('Preview must be a real boolean',()=>{
  assert.equal(parseChatBody({headers:{},body:{messages:[],preview:true}}).preview,true);
  assert.throws(()=>parseChatBody({headers:{},body:{messages:[],preview:'true'}}));
});
test('The local teacher contains no downloaded or pretrained runtime',()=>{
  const teacher=fs.readFileSync('trainer/teacher.py','utf8');
  assert.match(teacher,/generate_checked/);
  assert.doesNotMatch(teacher,/^import torch|^from transformers|snapshot_download|AutoModelForCausalLM/m);
  const starter=fs.readFileSync('trainer/start.py','utf8');
  assert.match(starter,/teacher=False/);
});
