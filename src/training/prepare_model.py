import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct'
    )

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct', 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
    )

# Adding special tokens
existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
GOOD_TOKEN='<+>'
BAD_TOKEN='<->'
SEPERATOR_TOKEN='<extra>'
new_special_tokens = [GOOD_TOKEN, BAD_TOKEN, SEPERATOR_TOKEN]
updated_special_tokens = list(set(existing_special_tokens + new_special_tokens))
tokenizer.add_special_tokens({"additional_special_tokens": updated_special_tokens})

# Editing embedding of SEPERATOR_TOKEN
with torch.no_grad():
    extra_token_id = tokenizer.convert_tokens_to_ids(SEPERATOR_TOKEN)
    base_token_id = tokenizer.pad_token_id  # or use "<mask>", "unused", etc.
    model.get_input_embeddings().weight[extra_token_id] = model.get_input_embeddings().weight[base_token_id]


save_directory = "./models/Qwen2.5-Math-7B-Instruct-updated"

os.makedirs(save_directory, exist_ok=True)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)