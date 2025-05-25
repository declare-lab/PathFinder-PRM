from datasets import load_dataset
from math_verify import parse, verify
import json
import argparse


parser = argparse.ArgumentParser(description="Greedy Search results evalaution")

parser.add_argument("--results_dir", type=str, required=True, help="Directory with saved results")
args = parser.parse_args()

results_dir = args.results_dir

def verify_ans(gold, answer):
    try:
        gold = parse("$"+gold+"$")
        answer = parse("$"+answer+"$")
    except Exception as e:
        print("Error Parsing: ", e)
        print(gold, answer)
    try:
        output = verify(gold, answer)
        return output
    except BaseException as e:
        print("Error Comparing: ", e)
        print(gold, answer)

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
        return s

datasets = ["math", "amc23",  "aime24", "college_math", "olympiadbench", "minerva_math"]
results = {}
score_sum = 0

for dataset in datasets:
    ds = load_dataset("json", 
                    data_files=f"{results_dir}/{dataset}/result-0-None.json",
                    split = "train")
    ds_filtered = ds.filter(lambda x: x['pred_answer']!=None)

    print(len(ds), len(ds_filtered))
    if dataset != "minerva_math":
        score = sum([
            verify_ans(str(ds_filtered[idx]["gt_answer"]), ds_filtered[idx]["pred_answer"]) \
                for idx in range(len(ds_filtered))
            ])/len(ds)
    else:
        score = sum([
            verify_ans(extract_boxed(str(ds_filtered[idx]["gt_answer"])), ds_filtered[idx]["pred_answer"]) \
                for idx in range(len(ds_filtered))
            ])/len(ds)
        
    results[dataset] = score
    score_sum += score
        

    with open(f"{results_dir}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

avg_score = score_sum/len(datasets)
results['average'] = avg_score

with open(f"{results_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)