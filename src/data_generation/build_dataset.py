import json
import time
from tqdm import tqdm  # Import tqdm for progress bar

GOOD_TOKEN='<+>'
BAD_TOKEN='<->'
SEPARATOR_TOKEN='<extra>'
SYSTEM_PROMPT="You are a Math Teacher. Given a question and a student's solution, evaluate the mathematical correctness, logic consistency of the current step and whether it will lead to the correct final solution"

def build_data_sample(sample):
    if len(sample['prev_steps'])>0:
        asst_prompt = sample['prev_steps'] + "\n\nCurrent Step: " + sample['curr_step']
    else:
        asst_prompt = "Current Step: " + sample['curr_step']

    if sample.get('discarded', False):
        return None
    if sample.get('score_A', -1) == -1:
        return None
    try:
        token_A = GOOD_TOKEN if sample['score_A'] == 1 else BAD_TOKEN
        token_B = GOOD_TOKEN if sample['score_B'] == 1 else BAD_TOKEN
        token_O = GOOD_TOKEN if sample['score_A'] == 1 and sample['score_B'] == 1 and sample['score_C'] == 1 else BAD_TOKEN  
    except KeyError as e:
        print("Missing key:", e)
        print("Offending sample:", sample)
        raise
        return None
    
    sample_1 = asst_prompt + f" Math reasoning: {token_A}, Consistency: {token_B}"
    # 4th label is dependent on prev 3 labels
    sample_2 = asst_prompt + f" Math reasoning: {token_A}, Consistency: {token_B}, Correctness: {token_O}"
    
    inp_1 = asst_prompt + f" Math reasoning: {SEPARATOR_TOKEN}, Consistency: {SEPARATOR_TOKEN}"
    # 4th label is dependent on prev 3 labels
    inp_2 = asst_prompt + f" Math reasoning: {token_A}, Consistency: {token_B}, Correctness: {SEPARATOR_TOKEN}"
    
    samples = [sample_1, sample_2]
    inputs = [inp_1, inp_2]
    
    return build_conversation(sample['question'], samples, inputs)

def build_conversation(user_prompt, samples, inputs):
    data = []
    for i in range(len(inputs)):
        data.append(
            {'labels': [
                {
                    "role":"user", "content": SYSTEM_PROMPT + "\n\n Question: "+ user_prompt
                },
                {
                    "role":"assistant", "content": samples[i]
                }
            ],
            'inputs': [
                {
                    "role":"user", "content": SYSTEM_PROMPT + "\n\n Question: "+ user_prompt
                },
                {
                    "role":"assistant", "content": inputs[i]
                },
            ],
            },
        )
    return data

print("Loading dataset...")

with open('./dataset/Disc_PRM_ds_1.json', 'r') as f:
    prm800k_1 = json.load(f)

with open('./dataset/Disc_PRM_ds_2.json', 'r') as f:
    prm800k_2 = json.load(f)

with open('./dataset/processed_mistral_dataset_scored_cleaned_1.json', 'r') as f:
    mistral_1 = json.load(f)

with open('./dataset/processed_mistral_dataset_scored_cleaned_2.json', 'r') as f:
    mistral_2 = json.load(f)

scored_dataset = prm800k_1 + prm800k_2 + mistral_1 + mistral_2

print(len(scored_dataset))
input("Start:")
# For testing with a smaller subset, uncomment this line
# scored_dataset = scored_dataset[:100]


new_dataset = []

total_samples = len(scored_dataset)
start_time = time.time()
processed = 0
valid_samples = 0

for sample in tqdm(scored_dataset, desc="Processing samples", unit="sample"):
    processed += 1
    
    new_sample = build_data_sample(sample)
    
    if new_sample is not None:
        new_dataset.extend(new_sample)
        valid_samples += 1
    
    if processed % 1000 == 0:
        elapsed_time = time.time() - start_time
        samples_per_second = processed / elapsed_time
        estimated_total_time = total_samples / samples_per_second
        remaining_time = estimated_total_time - elapsed_time
        
        hours, remainder = divmod(remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nProgress: {processed}/{total_samples} samples processed ({valid_samples} valid)")
        print(f"Speed: {samples_per_second:.2f} samples/second")
        print(f"ETA: {int(hours)}h {int(minutes)}m {int(seconds)}s remaining\n")

print(f"\nProcessing complete! {processed} samples processed, {valid_samples} valid samples generated.")
print(f"Total time: {(time.time() - start_time):.2f} seconds")

filename = 'Disc_PRM_full_dataset.json'

print(len(new_dataset))
input("Save?")

try:
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        json.dump(new_dataset, f, indent=4)
    print(f"Data successfully saved to '{filename}'")
except IOError as e:
    print(f"Error writing to file '{filename}': {e}")

from datasets import Dataset
hf_dataset = Dataset.from_list(new_dataset)
split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=42)

split_dataset.save_to_disk("./dataset/Disc_PRM_full_dataset")
