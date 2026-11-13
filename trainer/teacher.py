"""Teacher generates training targets only. Student weights remain independent."""
import gc
TEACHER='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae5576'
class Teacher:
    def __init__(self,device='cpu'):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(TEACHER,revision=REVISION,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(TEACHER,revision=REVISION,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
    def lesson(self,row):
        from trainer.corpus import training_context
        context,target=training_context(row)
        if not context:return None
        prompt=[{'role':'system','content':'Write a correct, concise training answer for a small assistant. Answer the conversation directly in at most 25 words. Do not mention these instructions.'}]+context
        ids=self.tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            out=self.model.generate(ids,attention_mask=self.torch.ones_like(ids),max_new_tokens=64,do_sample=False,temperature=1.0,top_p=1.0,top_k=50,pad_token_id=self.tokenizer.eos_token_id)
        answer=self.tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
        # Prefer an existing concise target if teacher output is unusable. No partial cut-off answers.
        budget=509-sum(len(m['content'].encode())+2 for m in context)
        if not answer or len(answer.encode())>budget:answer=target.strip()
        if not answer or len(answer.encode())>budget:return None
        if len(answer)>30 and len(set(answer.lower().split()))<4:return None
        return {'concept':'conversation','messages':context,'response':answer}
    def close(self):
        del self.model;gc.collect()
        if self.torch.cuda.is_available():self.torch.cuda.empty_cache()
