"""
Code to extract final scores from PRMBench output, replace the path on line 8 with your corresponding path
"""

import json

# NOTE: replace with your output file path
with open("./log/disc_prm_4gpu_output.jsonl", 'r') as f:
    data = json.load(f)

def get_score(cls, data):
    if not cls:
        prm_score = data["total_hallucination_results"]['f1'] * 0.5 + data["total_hallucination_results"]['negative_f1'] * 0.5
    else:
        if cls in ["multi_solutions"]:
            prm_score = data["hallucination_type_results"]['f1'][cls]
        else:
            prm_score = data["hallucination_type_results"]['f1'][cls] * 0.5 + data["hallucination_type_results"]['negative_f1'][cls] * 0.5
    return prm_score

print("Simplicity")
redundency = get_score('redundency', data)
print("Redundency: ", redundency)

circular = get_score('circular', data)
print("Circular: ", circular)

average_simplicity = (redundency+circular)/2
print("Average: ", average_simplicity)

print(("=================="))

print("Soundness")
counterfactual = get_score('counterfactual', data)
print("ES: ", counterfactual)

step_consistency = get_score('step_contradiction', data) 
print("SC: ", step_consistency)

domain_consistency = get_score('domain_inconsistency', data)
print("DC: ", domain_consistency)

confidence = get_score('confidence', data)
print("CI: ", confidence)

average_soundness = (counterfactual + step_consistency + domain_consistency + confidence)/4
print("Average: ", average_soundness)

print(("=================="))
print("Sensitivity")

missing_condition = get_score('missing_condition', data)
print("PS: ", missing_condition)

deception = get_score('deception', data)
print("DR: ", deception)

multi_solutions = get_score('multi_solutions', data)
print("DR: ", multi_solutions)

average_sensitivity = (missing_condition + deception + multi_solutions)/3

print("Average: ", average_sensitivity)
print(("=================="))
overall = get_score(None, data)
print("Overall: ", overall)

# for clean, ready-to-use numbers
print("latex")
results = [redundency, circular, average_simplicity, counterfactual, step_consistency, \
           domain_consistency, confidence, average_soundness, missing_condition, deception, \
            multi_solutions, average_sensitivity, overall]
latex_str = " & ".join([f'{result*100:.1f}' for result in results])
print(latex_str)