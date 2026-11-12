import {quality} from '../server/train-store.js';
import {noCache,method,sendError} from '../server/train-http.js';
export default async function handler(req,res){noCache(res);if(!method(req,res,'GET'))return;
 try{return res.status(200).json(await quality());}catch(e){return sendError(res,e)} }
