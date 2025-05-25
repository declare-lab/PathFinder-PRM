import os
import argparse
import json
from typing import List
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI
from PathFinderPRM import PathFinderPRM
from prompts.policy_prompt import Qwen_Policy_Prompt

import regex

# Define the allowed character set:
# - \p{Latin} matches all Latin letters
# - \p{Greek} matches all Greek letters
# - \d matches digits
# - \s matches whitespace characters
# - The following are common math symbols, punctuation, and symbols frequently used in LaTeX
allowed_pattern = regex.compile(
    r"[^\p{Latin}\p{Greek}\d\s"
    r"\+\-\*/=\^\_\'\".,:;?!\(\)\{\}\[\]\\\$%<>|&@#"
    r"√∞±×÷°]"
)

SYSTEM_PROMPT="You are a Math Teacher. Given a question and a student's solution, evaluate the mathemetical correctness, logic consistency of the current step and whether it will lead to the correct final solution"
EOS_TOKEN="<|im_end|>"

def extract_boxed(s):
    results = []
    i = 0
    while True:
        start = s.find(r'\boxed{', i)
        if start == -1:
            break
        # advance past the “\boxed{”
        j = start + len(r'\boxed{')
        depth = 1
        while j < len(s) and depth > 0:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            # everything from just after the first '{' up to j-1
            content = s[start + len(r'\boxed{') : j - 1]
            results.append(content)
            i = j
        else:
            # unbalanced braces: bail out
            break
    if len(results) == 1:
        return results[0]
    else:
        return None


def find_illegal_chars(text: str) -> list:
    """
     Find characters in the text that are outside the allowed set, returning a list.
    """
    return allowed_pattern.findall(text)

def is_math_answer_valid(answer: str) -> bool:
    """
    Check whether the math answer contains illegal characters:
      - If returns True, it means the text has no disallowed characters
      - If returns False, the text contains illegal characters
    """
    illegal = find_illegal_chars(answer)
    if illegal:
        return False
    return True

def get_next_step(policy_model, question, previous_steps, policy_prompt, temperature):
    prompt = policy_prompt
    previous_step = "" if len(previous_steps) == 0 else "\n\n".join(previous_steps)
    now_prompt = prompt.format(question=question, previous_steps=previous_step)
    now_prompt = now_prompt.replace("left-brackets", "{").replace("right-brackets", "}")

    if len(previous_steps)!=0:
       now_prompt += "\n\n"

    sampling_params = SamplingParams(
        n=8,
        top_p=0.95,
        temperature=temperature,
        max_tokens=512,
        stop=[EOS_TOKEN, "\n\n"],
        include_stop_str_in_output=True,
    )
    llm_outputs = policy_model.generate(now_prompt, sampling_params)

    new_steps_candidates = []
    str_set = set()
    for output_obj in llm_outputs[0].outputs:
        gen_text = output_obj.text
        if not is_math_answer_valid(gen_text):
            continue
        if gen_text.endswith("\n\n"):
            gen_text = gen_text[:-2].strip()
        else:
            gen_text += EOS_TOKEN
        gen_text =gen_text

        if gen_text in str_set:
            continue
        new_steps_candidates.append(gen_text)
        str_set.add(gen_text)
    return new_steps_candidates


def get_reward(reward_model: PathFinderPRM, question: str, previous_steps: List[str], now_step: str) -> float:
    reward_prompt = [
        {"role":"user", "content": SYSTEM_PROMPT + "\n\n Question: "+ question}
    ]
    if len(previous_steps) > 0:
        prev_steps_str = "\n\n".join(previous_steps)
        reward_prompt.append(
            {"role": "user", "content": prev_steps_str + "\n\nCurrent Step: " + now_step +" Math reasoning: <extra>, Consistency: <extra>"}
        )
    else:
        reward_prompt.append(
            {"role": "user", "content": "Current Step: " + now_step +" Math reasoning: <extra>, Consistency: <extra>"}
        )
    reward_score = reward_model.inference_single(reward_prompt)

    return reward_score


def main():
    parser = argparse.ArgumentParser(description="Greedy Search reasoning pipeline with reward model")

    parser.add_argument("--policy_model_path", type=str, required=True, help="Path to the policy model.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model.")

    parser.add_argument("--data", type=str, required=True, help="Dataset to Evaluate on", 
                        choices = ["math", "amc23", "aime25", "aime24", "college_math", "minerva_math", "olympiadbench"])

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="the temperature of the policy model.")
    parser.add_argument("--data_begin", type=int, default=0, help="Starting index of the dataset to process.")
    parser.add_argument("--data_end", type=int, default=None, help="Ending index of the dataset to process.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DATA_PATHS = {
        "math": "./eval_data/MATH/Math-OAI.jsonl", 
        "amc23": "./eval_data/AMC23/test.jsonl", 
        "aime25": "./eval_data/AIME25/AIME25_train.jsonl", 
        "aime24": "./eval_data/AIME24/test.jsonl",
        "college_math": "./eval_data/College_Math/college_math_200.jsonl", 
        "minerva_math": "./eval_data/Minerva-MATH/minerva-math.jsonl",
        "olympiadbench" : "./eval_data/OlympiadBench/olympiadbench_200.jsonl"
    }

    if args.data == "minerva_math":
        ans_key = "solution"
    elif args.data == "olympiadbench":
        ans_key = "final_answer"
    else:
        ans_key = "answer"

    dataset = load_dataset("json", data_files=DATA_PATHS[args.data], split="train")
    
    if args.data_begin!=0 or args.data_end != None:
        dataset = [dataset[i] for i in range(args.data_begin, args.data_end)]

    policy_model = LLM(
        model=args.policy_model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    
    reward_model_init_kwargs = {
        "torch_dtype":torch.bfloat16,
        "attn_implementation":"flash_attention_2",
        # "device_map":"auto"
        "device_map":"cuda:1"
    }

    reward_model = PathFinderPRM(args.reward_model_path, reward_model_init_kwargs)

    new_dataset = []
    for i, data in tqdm(enumerate(dataset), desc="Processing dataset", total=len(dataset)):

        question = data["problem"]
        previous_steps = []
        used_steps = set()
        max_iteration = 30
        iteration_history = []
        iteration_index = 0

        while max_iteration > 0:
            max_iteration -= 1
            iteration_index += 1

            new_steps_candidates = get_next_step(policy_model, question, previous_steps, Qwen_Policy_Prompt, args.temperature)
            if len(new_steps_candidates) == 0:
                print(f"Question Number: {i}, No candidate generated at iteration {iteration_index}.")
                continue
            print(f"Question Number: {i}, Iteration: {iteration_index}, New Steps Candidates Number: {len(new_steps_candidates)}")
            new_step_accepted = False
            iteration_data = {
                "iteration_index": iteration_index,
                "candidates_info": [],
                "chosen_step_index": None,
                "chosen_step": None
            }

            for candidate_idx, candidate_step in enumerate(new_steps_candidates):
                if candidate_step in used_steps:
                    continue
                if candidate_step.endswith(EOS_TOKEN):
                    boxed_answer = extract_boxed(candidate_step)
                    if boxed_answer == None:
                        print(candidate_step)
                        continue
                    temp_candidate_step = candidate_step.replace(EOS_TOKEN, "")
                else:
                    temp_candidate_step = candidate_step
                
                
                reward_score = get_reward(
                    reward_model, question, previous_steps, temp_candidate_step
                )
                if reward_score == -1:
                    continue
                reward = reward_score
                
                print(f"Question Number: {i}, Previous Steps Number: {len(previous_steps)}, Candidate Step Index: {candidate_idx}, Reward: {reward}")

                candidate_info = {
                    "candidate_step": candidate_step,
                    "reward": reward
                }
                iteration_data["candidates_info"].append(candidate_info)
            
            max_reward = -1
            max_reward_idx = -1
            for idx, candidate_info in enumerate(iteration_data["candidates_info"]):
                if candidate_info["reward"] > max_reward:
                    max_reward = candidate_info["reward"]
                    max_reward_idx = idx
            if max_reward_idx == -1:
                continue
            iteration_data["chosen_step_index"] = max_reward_idx
            iteration_data["chosen_step"] = iteration_data["candidates_info"][max_reward_idx]["candidate_step"]
            previous_steps.append(iteration_data["chosen_step"])
            iteration_history.append(iteration_data)

            if len(previous_steps) > 0 and EOS_TOKEN in previous_steps[-1]:
                print(f"Question Number: {i}, Early stopping at iteration {iteration_index}.")
                break

            # # If allow backtracking (not Greedy)
            # if not new_step_accepted:
            #     if previous_steps:
            #         previous_steps.pop()
            #     print(f"Question Number: {i}, No new step accepted at iteration {iteration_index}. Retrying previous step.")

        new_dataset.append({
            "question": question,
            "iteration_history": iteration_history,
            "final_steps": previous_steps,
            "gt_answer": data[ans_key],
            "pred_answer": extract_boxed(previous_steps[-1]),
        })

        output_file = os.path.join(args.output_dir, f"result-{args.data_begin}-{args.data_end}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"Done! Results are saved to {output_file}.")


if __name__ == "__main__":
    main()