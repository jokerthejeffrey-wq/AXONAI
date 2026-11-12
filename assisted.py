"""Temporary pretrained inference. Never copies weights into the student."""
from settings import TEACHER,TEACHER_REVISION
from identity import COMMUNITY
SYSTEM=(f'You are AXONAI, a community AI project. {COMMUNITY} '
        'This chat uses the pretrained Qwen model developed by the Qwen team at Alibaba, while an independent AXONAI student is being trained. '
        'Distinguish the community application from the pretrained underlying model when asked. '
        'Do not claim AXONAI was created by Anthropic or OpenAI. Be helpful, concise, honest about uncertainty, and use the conversation context. '
        'You do not learn permanently from these chat messages. Do not invent a last-update date.')
class AssistedModel:
    def __init__(self,device='cpu'):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(2);self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(TEACHER,revision=TEACHER_REVISION,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(TEACHER,revision=TEACHER_REVISION,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32).to(device).eval()
    def reply(self,messages):
        import torch
        history=[{'role':'system','content':SYSTEM}]+messages
        while True:
            ids=self.tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,return_tensors='pt')
            if ids.shape[1]<=1800:break
            if len(history)>2:history=[history[0]]+history[3:]
            else:
                history[-1]=dict(history[-1],content=history[-1]['content'][-2000:])
                ids=self.tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,return_tensors='pt')
                if ids.shape[1]>1800:raise RuntimeError('Message too long for assisted context. Please shorten it.')
                break
        ids=ids.to(self.device)
        with torch.inference_mode():
            out=self.model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=True,temperature=.65,top_p=.9,repetition_penalty=1.08,pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
