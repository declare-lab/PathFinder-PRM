import json
import re
from pathlib import Path

# NOTE: for this program to work, please download the mistral dataset as .json from https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data and store it under ./dataset

"""
Launch with: python load_data_mistral.py dataset/mistral_dataset.json
"""

def load_mistral_dataset(file_path):
    """Load the Mistral dataset from a JSON file"""
    try:
        print(f"Loading dataset from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Check if it's a list of dictionaries with 'conversations' key
        if isinstance(dataset, list):
            print(f"Loaded dataset with {len(dataset)} entries")
            
            # Look at the first element to determine format
            if len(dataset) > 0 and isinstance(dataset[0], dict) and 'conversations' in dataset[0]:
                print(f"Format: List of dictionaries with 'conversations' key")
                # Convert to the format needed for processing (extract conversations)
                conversations = []
                for item in dataset:
                    if 'conversations' in item and isinstance(item['conversations'], list):
                        conversations.append(item['conversations'])
                
                print(f"Extracted {len(conversations)} conversation sequences")
                return conversations
            else:
                print(f"Format: List of examples")
                return dataset
        else:
            print(f"WARNING: Unexpected dataset format. Expected a list but got {type(dataset).__name__}")
            return []
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return []

def process_mistral_dataset(dataset, output_folder="./dataset/"):
    """
    Process the Mistral dataset into the desired format.
    Each record contains:
    - question: just the question text
    - prev_steps: previous steps as a string
    - curr_step: the current step text
    - context: formatted string with question, prev_steps, and curr_step
    - mistral_score: 1 for '+', 0 for '-'
    - score_A, score_B, score_C: all initialized to -1
    - detailed_context: list of previous steps with metadata
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "processed_mistral_dataset.json"
    
    processed_records = []
    skipped_examples = 0
    
    for example_idx, example in enumerate(dataset):
        try:
            # Validate example structure
            if not example or not isinstance(example, list) or len(example) < 2:
                print(f"Skipping example {example_idx}: Invalid structure")
                skipped_examples += 1
                continue
                
            # Check if first message has required fields
            if not isinstance(example[0], dict) or "content" not in example[0] or "role" not in example[0]:
                print(f"Skipping example {example_idx}: Missing required fields in first message")
                skipped_examples += 1
                continue
            
            # Extract question from the first user message
            first_message = example[0]["content"]
            match = re.match(r"(.*?)Step 1:(.*)", first_message, re.DOTALL)
            if match:
                question_text = match.group(1).strip()
                first_step_text = "Step 1:" + match.group(2).strip()
            else:
                # Fallback if pattern doesn't match
                question_text = first_message
                first_step_text = ""
            
            # Initialize for processing steps
            previous_steps = []
            detailed_previous_steps = []
            
            # Process each step (every other message, starting from index 0)
            for i in range(0, len(example), 2):
                try:
                    # Get the step text
                    if i == 0:
                        step_text = first_step_text
                    else:
                        if "content" not in example[i]:
                            print(f"Skipping step in example {example_idx}: Missing content field")
                            continue
                        step_text = example[i]["content"]
                    
                    # Get the rating from the assistant message that follows
                    mistral_score = -1  # Default if no rating available
                    if i + 1 < len(example):
                        if "content" not in example[i+1]:
                            print(f"Skipping rating in example {example_idx}: Missing content field")
                        else:
                            rating = example[i+1]["content"]
                            mistral_score = 1 if rating == "+" else 0
                    
                    # Build context
                    context = ""
                    if previous_steps:
                        context = "\n\n".join(previous_steps)
                        
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
                        "mistral_score": mistral_score,  
                        "score_A": -1,
                        "score_B": -1,
                        "score_C": -1,
                        "detailed_context": detailed_previous_steps.copy()
                    }
                    processed_records.append(record)
                    

                    previous_steps.append(step_text)
                    
                    detailed_previous_steps.append({
                        "step": len(detailed_previous_steps) + 1,
                        "step_text": step_text,
                        "mistral_score": mistral_score 
                    })
                except Exception as e:
                    print(f"Error processing step {i} in example {example_idx}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing example {example_idx}: {e}")
            skipped_examples += 1
            continue
    
    # Write processed records to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_records)} records, saved to {output_file}")
    print(f"Skipped {skipped_examples} examples due to errors")
    return processed_records


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python load_data_mistral.py <path_to_dataset.json>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    dataset = load_mistral_dataset(file_path)
    if dataset:
        processed_data = process_mistral_dataset(dataset)
    else:
        print("No data to process. Exiting.")
        sys.exit(1)

