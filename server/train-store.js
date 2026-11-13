import { neon } from '@neondatabase/serverless';
import { put } from '@vercel/blob';
import crypto from 'node:crypto';
import {CHUNK_BYTES,MAX_CHECKPOINT,BATCH,MANIFEST,makeLessons,sha256,packChunks,mergeWeights,workerOK,idOK,tokenOK} from './train-core.js';
import fs from 'node:fs';
import path from 'node:path';
import {initialCheckpoint} from './base-checkpoint.js';
import {parseCheckpoint,scoreNextByte} from './scratch-infer.js';
export const PENDING_QUEUE_LIMIT=12;
class InvalidCandidate extends Error {}

let initialized;
export class TrainError extends Error {constructor(message,status=400){super(message);this.status=status;}}
export function configured(){return !!(process.env.DATABASE_URL && (process.env.BLOB_READ_WRITE_TOKEN || (process.env.BLOB_STORE_ID && process.env.VERCEL_OIDC_TOKEN)));}
const check = (condition,message,status=400)=>{if(!condition)throw new TrainError(message,status)};
function connection(){check(configured(),'Training storage is not configured.',503);return neon(process.env.DATABASE_URL);}
async function setup(sql){
  // Calls are safe if multiple serverless instances start at the same time.
  await sql`CREATE TABLE IF NOT EXISTS axon_meta (key text primary key, val bigint not null)`;
  await sql`INSERT INTO axon_meta(key,val) VALUES ('version',0),('sequence',0),('corpus_sequence',0) ON CONFLICT (key) DO NOTHING`;
  await sql`CREATE TABLE IF NOT EXISTS axon_jobs (
    id text primary key, worker text not null, version bigint not null, issued_at timestamptz not null default now(),
    secret_hash text not null, status text not null default 'issued', rows jsonb not null,
    size bigint, sha256 text, submitted_at timestamptz, merged_version bigint,
    constraint axon_jobs_status CHECK(status IN ('issued','pending','approved','rejected'))
  )`;
  await sql`ALTER TABLE axon_jobs ADD COLUMN IF NOT EXISTS corpus jsonb`;
  await sql`CREATE TABLE IF NOT EXISTS axon_parts (
    job_id text not null references axon_jobs(id), part_index integer not null,
    url text not null, size integer not null, primary key(job_id,part_index)
  )`;
  await sql`CREATE TABLE IF NOT EXISTS axon_checkpoints (
    version bigint primary key, parts jsonb not null, size bigint not null, sha256 text not null
  )`;
  await sql`CREATE TABLE IF NOT EXISTS axon_daily (day date primary key, claims integer not null default 0)`;
  await sql`CREATE TABLE IF NOT EXISTS axon_queue_lease (id integer primary key, owner text not null, locked_until timestamptz not null)`;
  await sql`INSERT INTO axon_queue_lease (id,owner,locked_until) VALUES (1,'',to_timestamp(0)) ON CONFLICT DO NOTHING`;
  await sql`CREATE TABLE IF NOT EXISTS axon_quality (
    version bigint primary key, score integer, passed integer, total integer, tested boolean not null default false,
    ready boolean not null default false, checked_at timestamptz
  )`;
}
export async function db(){const sql=connection();if(!initialized){initialized=setup(sql).catch(e=>{initialized=undefined;throw e});}await initialized;return sql}
async function currentVersion(sql){const [v]=await sql`SELECT val FROM axon_meta WHERE key='version'`;return Number(v?.val??0)}
export async function publicStatus(){
  if(!configured())return {ok:true,training_available:false,version:0,reserved_lessons:0,pending_updates:0,reason:'storage_not_connected'};
  try{
    const sql=await db();
    const [v]=await sql`SELECT val FROM axon_meta WHERE key='version'`;
    const [l]=await sql`SELECT coalesce(sum(val),0) AS val FROM axon_meta WHERE key IN ('sequence','corpus_sequence')`;
    const [p]=await sql`SELECT count(*)::integer AS n FROM axon_jobs WHERE status='pending'`;
    return {ok:true,training_available:true,version:Number(v?.val||0),reserved_lessons:Number(l?.val||0),pending_updates:Number(p?.n||0),queue_capacity:PENDING_QUEUE_LIMIT,queue_full:Number(p?.n||0)>=PENDING_QUEUE_LIMIT};
  }catch(e){
    // Log only a bounded, redacted diagnostic; do not log the connection
    // string, credentials, request bodies, or SQL parameters.
    const sqlState=String(e?.code||e?.cause?.code||'unknown').replace(/[^a-zA-Z0-9_-]/g,'').slice(0,30);
    const detail=String(e?.message||'unknown')
      .replace(/(?:postgres(?:ql)?):\/\/[^\s'"]+/gi,'[database-url]')
      .replace(/(?:password|token|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+/gi,'[credential]')
      .slice(0,220);
    console.error('Training DB status failed',e?.name,sqlState,detail);
    return {ok:true,training_available:false,version:0,reason:'storage_unreachable'}
  }
}
export async function quality(){
  const status=await publicStatus();
  if(!status.training_available)return {tested:false,ready:false,score:null,version:0,review:'pending'};
  const sql=await db();const [q]=await sql`SELECT * FROM axon_quality WHERE version=${status.version}`;
  return {tested:!!q?.tested,ready:false,score:q?.tested?Number(q.score):null,version:status.version,
    passed:q?.passed??0,total:q?.total??0,checked_at:q?.checked_at??null,review:q?.tested?'reviewed':'pending'};
}
export async function claim(worker,corpus=null){
  if(corpus)check(corpus.id==='smoltalk-v1'&&corpus.revision==='2ca82d2fe63c017af8d4cb30e81e95cb25516851'&&Number.isSafeInteger(corpus.records)&&corpus.records>=8&&corpus.records<=1_000_000,'Unsupported corpus');
  check(workerOK(worker),'Invalid worker identifier');
  const sql=await db();
  let [pending]=await sql`SELECT count(*)::integer AS n FROM axon_jobs WHERE status='pending'`;
  if(Number(pending.n)>0){
    try{await autoReviewOne(sql)}
    catch(error){console.error('AXON queue verification deferred:',String(error?.name||'Error').slice(0,45))}
    ;[pending]=await sql`SELECT count(*)::integer AS n FROM axon_jobs WHERE status='pending'`;
    // Older trainers will stop cleanly on 204 instead of looping on a 429.
    if(Number(pending.n)>=PENDING_QUEUE_LIMIT)return null;
  }
  await sql`INSERT INTO axon_daily(day,claims) VALUES(CURRENT_DATE,0) ON CONFLICT DO NOTHING`;
  const [quota]=await sql`UPDATE axon_daily SET claims=claims+1 WHERE day=CURRENT_DATE AND claims<96 RETURNING claims`;
  if(!quota)return null; // All daily lessons reserved: allow old trainers to exit without an endless 429 loop.
  const [v]=await sql`SELECT val FROM axon_meta WHERE key='version'`;
  const key=corpus?'corpus_sequence':'sequence';
  const [seq]=await sql`UPDATE axon_meta SET val=val+${BATCH} WHERE key=${key}
    AND val+${BATCH}<=${corpus?corpus.records:Number.MAX_SAFE_INTEGER} RETURNING val-${BATCH} AS start`;
  if(!seq)return null;
  const id=crypto.randomBytes(16).toString('hex'),secret=crypto.randomBytes(32).toString('hex');
  const assignment=corpus?{id:corpus.id,revision:corpus.revision,start:Number(seq.start),count:BATCH}:null;
  const rows=assignment?[]:makeLessons(Number(seq.start));
  await sql`INSERT INTO axon_jobs(id,worker,version,secret_hash,rows,corpus) VALUES(${id},${worker},${Number(v.val)},${sha256(Buffer.from(secret))},${JSON.stringify(rows)}::jsonb,${JSON.stringify(assignment)}::jsonb)`;
  return {id,version:Number(v.val),manifest:MANIFEST,rows,corpus:assignment,upload_token:secret};
}
async function authorizedJob(sql,id,secret){
  check(idOK(id)&&tokenOK(secret),'Invalid job authorization',401);
  const [j]=await sql`SELECT * FROM axon_jobs WHERE id=${id}`;
  check(j&&j.secret_hash===sha256(Buffer.from(secret)),'Invalid job authorization',401);
  check(new Date(j.issued_at).getTime()>Date.now()-72*3600_000,'Job expired',409);
  return j;
}
export async function uploadPart(job,secret,index,buffer){
  check(Buffer.isBuffer(buffer)&&buffer.length>0&&buffer.length<=CHUNK_BYTES,'Invalid checkpoint chunk',413);
  check(Number.isInteger(index)&&index>=0&&index<16,'Invalid chunk index');
  const sql=await db();const j=await authorizedJob(sql,job,secret);
  check(j.status==='issued','Job is already submitted',409);
  const [existing]=await sql`SELECT url,size FROM axon_parts WHERE job_id=${job} AND part_index=${index}`;
  if(existing){check(Number(existing.size)===buffer.length,'Chunk already uploaded with different size',409);return {ok:true,part:index};}
  const blob=await put(`training/submissions/${job}/${index}-${crypto.randomBytes(6).toString('hex')}.bin`,buffer,
    {access:'public',contentType:'application/octet-stream',addRandomSuffix:false,cacheControlMaxAge:60});
  await sql`INSERT INTO axon_parts(job_id,part_index,url,size) VALUES(${job},${index},${blob.url},${buffer.length}) ON CONFLICT DO NOTHING`;
  return {ok:true,part:index};
}
export async function finish(job,secret,total,expectedHash,expectedSize){
  check(Number.isInteger(total)&&total>=1&&total<=16,'Invalid chunk count');
  check(typeof expectedHash==='string'&&/^[a-f0-9]{64}$/.test(expectedHash),'Invalid file checksum');
  check(Number.isSafeInteger(expectedSize)&&expectedSize>100&&expectedSize<=MAX_CHECKPOINT,'Invalid file size');
  const sql=await db(),j=await authorizedJob(sql,job,secret);
  check(j.status==='issued','Job is already submitted',409);
  const parts=await sql`SELECT part_index,size,url FROM axon_parts WHERE job_id=${job} ORDER BY part_index`;
  check(parts.length===total,'Missing uploaded chunks',409);
  check(parts.every((v,i)=>v.part_index===i),'Chunk sequence invalid',409);
  check(parts.reduce((a,p)=>a+Number(p.size),0)===expectedSize,'File size mismatch',409);
  // Check the complete checksum before accepting. The bytes are not executed.
  const uploaded=await assemble(parts);
  check(sha256(uploaded)===expectedHash,'File checksum mismatch',409);
  const [accepted]=await sql`UPDATE axon_jobs SET status='pending',size=${expectedSize},sha256=${expectedHash},submitted_at=now()
    WHERE id=${job} AND status='issued' RETURNING id`;
  check(!!accepted,'Job already submitted',409);
  return {id:job,status:'pending-owner-review'};
}
export async function manifestFor(version){
  check(Number.isSafeInteger(version)&&version>=0,'Invalid checkpoint version');
  if(version===0){
    return {version:0,...initialCheckpoint};
  }
  const sql=await db();const [row]=await sql`SELECT parts,size,sha256 FROM axon_checkpoints WHERE version=${version}`;
  check(!!row,'Checkpoint not found',404);
  return {version,parts:row.parts,size:Number(row.size),sha256:row.sha256};
}
export async function assemble(parts){
  const buffers=[],totalLimit=MAX_CHECKPOINT;let size=0;
  for(const part of parts){
    let url=part.url;
    if(url.startsWith('/model/')){
      const local=path.join(process.cwd(),'public',url.slice(1));
      if(fs.existsSync(local)){
        const buf=fs.readFileSync(local);buffers.push(buf);size+=buf.length;
      }else{
        // Vercel separates public static assets from function file bundles.
        const deployment=process.env.VERCEL_URL || process.env.VERCEL_PROJECT_PRODUCTION_URL;
        check(deployment,'Checkpoint host unavailable',503);
        const r=await fetch('https://'+deployment+url,{signal:AbortSignal.timeout(20000)});
        check(r.ok,'Checkpoint download unavailable',503);
        const data=Buffer.from(await r.arrayBuffer());buffers.push(data);size+=data.length;
      }
    }
    else{
      let parsed;try{parsed=new URL(url)}catch{throw new TrainError('Invalid blob URL',500)}
      check(parsed.protocol==='https:'&&parsed.username===''&&parsed.password===''&&parsed.hostname.endsWith('.public.blob.vercel-storage.com')&&parsed.port==='','Invalid blob host',500);
      const response=await fetch(url,{signal:AbortSignal.timeout(20000)});
      check(response.ok,'Checkpoint data unavailable',503);
      const ab=await response.arrayBuffer();size+=ab.byteLength;buffers.push(Buffer.from(ab));
    }
    check(size<=totalLimit,'Checkpoint exceeds maximum size',413);
  }
  return Buffer.concat(buffers);
}
export async function reviewList(owner){check(typeof owner==='string'&&owner===process.env.AXON_OWNER_TOKEN&&owner.length>=32,'Unauthorized',401);
  const sql=await db();const items=await sql`SELECT id,worker,version,status,size,issued_at,submitted_at,merged_version FROM axon_jobs ORDER BY issued_at DESC LIMIT 50`;
  return {submissions:items};
}
export async function review(owner,id,action){
  check(typeof owner==='string'&&owner===process.env.AXON_OWNER_TOKEN&&owner.length>=32,'Unauthorized',401);
  check(idOK(id),'Invalid job id');check(action==='approve'||action==='reject','Invalid action');
  const sql=await db();const [j]=await sql`SELECT * FROM axon_jobs WHERE id=${id}`;
  check(j&&j.status==='pending','Job not pending',409);
  if(action==='reject'){
    await sql`UPDATE axon_jobs SET status='rejected' WHERE id=${id} AND status='pending'`;
    return {id,status:'rejected'};
  }
  return integrateJob(sql,j,{automatic:false});

}


const VALIDATION_PROBES=[
  {messages:[{role:'user',content:'hello'}],expected:72},
  {messages:[{role:'user',content:'What is 2 + 2?'}],expected:52}
];
function validationScore(model){
  return VALIDATION_PROBES.map(probe=>scoreNextByte(model,probe.messages,probe.expected));
}
function validateCandidate(current,merged){
  let before,after;
  try{
    before=validationScore(parseCheckpoint(current));
    after=validationScore(parseCheckpoint(merged));
  }catch{throw new InvalidCandidate('Checkpoint validation failed')}
  if(after.some((loss,i)=>!Number.isFinite(loss)||loss>before[i]+0.85) ||
    (after[0]+after[1])>(before[0]+before[1])+0.55) {
    throw new InvalidCandidate('Checkpoint regresses on validation probes');
  }
}
async function integrateJob(sql,j,{automatic=false}={}){
  const id=j.id,version=await currentVersion(sql);
  const parts=await sql`SELECT part_index,size,url FROM axon_parts WHERE job_id=${id} ORDER BY part_index`;
  const [baseManifest,currentManifest]=await Promise.all([manifestFor(Number(j.version)),manifestFor(version)]);
  const [base,current,candidate]=await Promise.all([
    assemble(baseManifest.parts),assemble(currentManifest.parts),assemble(parts)
  ]);
  check(sha256(candidate)===j.sha256,'Submitted file checksum mismatch',400);
  let merged;
  try{merged=mergeWeights(base,current,candidate)}
  catch{throw new InvalidCandidate('Unsafe model weight update')}
  if(automatic)validateCandidate(current,merged);
  const next=version+1,urls=[];
  for(const [i,part] of packChunks(merged).entries()){
    const out=await put(`training/approved/v${next}/${i}-${crypto.randomBytes(6).toString('hex')}.bin`,part,{
      access:'public',contentType:'application/octet-stream',addRandomSuffix:false,cacheControlMaxAge:60
    });
    urls.push({url:out.url,size:part.length});
  }
  await sql`INSERT INTO axon_checkpoints(version,parts,size,sha256)
    VALUES(${next},${JSON.stringify(urls)}::jsonb,${merged.length},${sha256(merged)})
    ON CONFLICT DO NOTHING`;
  const [updated]=await sql`UPDATE axon_meta SET val=${next}
    WHERE key='version' AND val=${version} RETURNING val`;
  check(!!updated,'Checkpoint version changed during approval; retry.',409);
  await sql`UPDATE axon_jobs SET status='approved',merged_version=${next}
    WHERE id=${id} AND status='pending'`;
  return {id,status:'approved',version:next};
}
async function autoReviewOne(sql){
  // Serialize promotions across Vercel functions. Dead instances have a
  // time-limited lease so the queue can recover without a persistent server.
  const owner=crypto.randomBytes(18).toString('hex');
  const [lease]=await sql`UPDATE axon_queue_lease
    SET owner=${owner},locked_until=now()+interval '240 seconds'
    WHERE id=1 AND locked_until<now() RETURNING id`;
  if(!lease)return {processed:false,reason:'review_in_progress'};
  try{
    const [job]=await sql`SELECT * FROM axon_jobs
      WHERE status='pending' ORDER BY submitted_at ASC LIMIT 1`;
    if(!job)return {processed:false,reason:'empty'};
    try{
      const result=await integrateJob(sql,job,{automatic:true});
      return {processed:true,result};
    }catch(error){
      if(error instanceof InvalidCandidate ||
        (error instanceof TrainError && error.status>=400 && error.status<500 && error.status!==409)){
        await sql`UPDATE axon_jobs SET status='rejected'
          WHERE id=${job.id} AND status='pending'`;
        return {processed:true,result:{id:job.id,status:'rejected'}};
      }
      throw error;
    }
  }finally{
    await sql`UPDATE axon_queue_lease SET owner='',locked_until=to_timestamp(0)
      WHERE id=1 AND owner=${owner}`;
  }
}
