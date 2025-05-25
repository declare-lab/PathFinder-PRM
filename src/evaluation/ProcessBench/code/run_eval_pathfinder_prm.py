# example code to evaluate PathFinder-PRM

import argparse
import os
import numpy as np
import json
from tqdm import tqdm
import torch
from multiprocessing import Pool
from datasets import load_dataset

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT="You are a Math Teacher. Given a question and a student's solution, evaluate the mathematical correctness, logic consistency of the current step and whether it will lead to the correct final solution"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
        device_map = "auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    def inference_single(sample_input):
        model.eval()

        message_ids = tokenizer.apply_chat_template(
            sample_input,
            tokenize=True,
            return_dict=True,
            return_tensors='pt'
            ).to(model.device)

        mask_token_id = tokenizer.encode("<extra>")[0]
        token_masks = (message_ids['input_ids'] == mask_token_id)

        shifted_mask = torch.cat(
            [
                token_masks[:, 1:], 
                torch.zeros(token_masks.size(0), 1, dtype=torch.bool, device=model.device)
            ], 
            dim=1
            )

        with torch.no_grad():
            outputs = model(**message_ids)

        predicted_tokens = outputs.logits[shifted_mask].argmax(dim=-1)

        decoded_tokens = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in predicted_tokens]
        
        if not set(decoded_tokens).issubset({'<+>', '<->'}):
            raise ValueError(f"Invalid decoded labels {decoded_tokens}")

        return '<->' in decoded_tokens


    def single_process(d):
        steps = d['steps']
        messages = [
            {"role":"user", "content": SYSTEM_PROMPT + "\n\n Question: "+ d['problem']}
        ]
        for sdx, step in enumerate(steps):
            if sdx == 0:
                curr_messages = messages.copy()
                curr_messages.append(
                    {"role": "assistant", "content": "Current Step: "+step+" Math reasoning: <extra>, Consistency: <extra>"}
                )
                
            else:
                curr_messages = messages.copy()
                prev_steps = "\n\n".join(steps[:sdx])
                curr_messages.append(
                    {"role": "assistant", "content": prev_steps + "\n\nCurrent Step: "+step+" Math reasoning: <extra>, Consistency: <extra>"}
                )
            
            error_present = inference_single(curr_messages)

            if error_present:
                return sdx

        return -1

    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    for config in configs:
        input_data = load_dataset('Qwen/ProcessBench', split=config)
        
        predictions = list(tqdm(map(single_process, input_data), total=len(input_data),
                                desc=f'Processing {config}', dynamic_ncols=True))
        
        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d['prediction'] = predictions[idx]
            new_d['match'] = predictions[idx] == d['label']
            res_data.append(new_d)
        
        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]
        with open(f'{args.output_dir}/{config}_error.jsonl', 'w') as f:
            for e in data1:
                f.write(json.dumps(e) + '\n')
        with open(f'{args.output_dir}/{config}_correct.jsonl', 'w') as f:
            for e in data2:
                f.write(json.dumps(e) + '\n')
        
        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
        
        with open(f'{args.output_dir}/results.jsonl', 'a+') as f:
            f.write(json.dumps({
                "config": config,
                "error acc": acc1, 
                "correct acc": acc2,
                "f1": f1
            }) + '\n')

if __name__ == '__main__':
    main()
