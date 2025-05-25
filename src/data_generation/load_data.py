import json
from pathlib import Path

# NOTE: for this program to work, please download the four jsonl files below from https://github.com/openai/prm800k and store them under ./dataset

def load_prm800k_dataset(dataset_folder="./dataset"):
    dataset = []
    dataset_folder = Path(dataset_folder)

    files = [
        "phase1_train.jsonl",
        "phase1_test.jsonl",
        "phase2_train.jsonl",
        "phase2_test.jsonl"
    ]

    for file_name in files:
        file_path = dataset_folder / file_name
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))

    print(f"Total examples loaded: {len(dataset)}")
    return dataset

def create_new_dataset(dataset, dataset_folder="./dataset"):
    """
    Create a processed_dataset.json under dataset folder.
    Each record contains:
    - context: question + previous steps + this step text
    - question: question text
    - prev_steps: previous steps text
    - curr_step: this step text
    - human_score: original completion rating
    - score_A, score_B, score_C: all initialized to -1
    - detailed_context: list of previous steps with metadata
    """
    dataset_folder = Path(dataset_folder)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    out_path = dataset_folder / "processed_dataset.json"
    processed = []
    
    for example in dataset:
        question_text = example["question"]["problem"]
        steps = example["label"]["steps"]
        previous_steps = []
        detailed_previous_steps = [] 
        
        for step in steps:
            completions = step.get("completions") or []
            
            for completion in completions:
                context = ''
                if previous_steps:
                    context += "\n\n".join(previous_steps)
                    
                step_text = completion.get("text", "")
                full_context = f"""
                [Math Problem]
                {question_text}
                [Partial Solution]
                <previous_steps>
                {context}
                </previous_steps>
                <current_step>
                {step_text}
                </current_step>
                """
                
                record = {
                    "question": question_text,
                    "prev_steps": context,
                    "curr_step": step_text,
                    "context": full_context,
                    "human_score": completion.get("rating"),
                    "score_A": -1,
                    "score_B": -1,
                    "score_C": -1,
                    "detailed_context": detailed_previous_steps.copy()  
                }
                processed.append(record)
                
            chosen_idx = step.get("chosen_completion")
            chosen_text = ""
            chosen_score = None
            
            if chosen_idx is not None and 0 <= chosen_idx < len(completions):
                chosen_text = completions[chosen_idx].get("text", "")
                chosen_score = completions[chosen_idx].get("rating")
            elif step.get("human_completion"):
                chosen_text = step["human_completion"].get('text', "")
                chosen_score = step["human_completion"].get("rating")
            else:
                continue
                
            previous_steps.append(chosen_text)
            
            detailed_previous_steps.append({
                "step": len(detailed_previous_steps) + 1,
                "step_text": chosen_text,
                "human_score": chosen_score
            })
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    
    print(f"Processed dataset written to {out_path} ({len(processed)} records).")


if __name__ == "__main__":
    data = load_prm800k_dataset("./dataset")
    create_new_dataset(data)