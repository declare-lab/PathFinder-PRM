import json
import re

def clean_prev_and_curr_steps(json_list):
    """
    Removes 'Step X:' labels from 'prev_steps' and 'curr_step' fields
    in each JSON object in the list.
    """
    step_pattern = re.compile(r"Step\s*\d+:\s*")

    for item in json_list:
        # Clean prev_steps
        if 'prev_steps' in item and isinstance(item['prev_steps'], str):
            lines = item['prev_steps'].strip().split('\n')
            cleaned_prev = [step_pattern.sub('', line).strip() for line in lines if line.strip()]
            item['prev_steps'] = '\n\n'.join(cleaned_prev)

        # Clean curr_step
        if 'curr_step' in item and isinstance(item['curr_step'], str):
            lines = item['curr_step'].strip().split('\n')
            cleaned_curr = [step_pattern.sub('', line).strip() for line in lines if line.strip()]
            item['curr_step'] = '\n\n'.join(cleaned_curr)

    return json_list

# NOTE: modify according to your filenames
input_path = './dataset/processed_mistral_dataset_scored.json'
output_path = './dataset/processed_mistral_dataset_scored_cleaned.json'

with open(input_path, 'r') as f:
    mistral_1 = json.load(f)

mistral_1 = clean_prev_and_curr_steps(mistral_1)

with open(output_path, 'w') as f:
    json.dump(mistral_1, f, indent=2)

print(f"Cleaned dataset saved to: {output_path}")