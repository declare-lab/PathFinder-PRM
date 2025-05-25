import time
import torch # type: ignore
from typing import Tuple, Any, List
import json
from pathlib import Path
import re
import gc
import os
from transformers import AutoTokenizer
from prompt import CLS_PROMPT
from vllm import LLM, SamplingParams # type: gnore
from tqdm import tqdm
from datetime import datetime, timedelta

# Configs
# NOTE: Please adjust these configs to your respective hardware setup

BATCH_SIZE = 1024  
os.environ["VLLM_NUM_SCHEDULER_STEPS"] = "10"
os.environ["VLLM_USE_MODELSCOPE"] = "False"
os.environ["VLLM_FUSED_MOE"] = "1"  # If using MoE models
os.environ["VLLM_USE_CONTINUOUSBATCHING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

class VLLMInferenceWithRetry:
    def __init__(
        self, 
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
        max_retries: int = 3,
        tensor_parallel_size: int = 8, 
        batch_size: int = BATCH_SIZE,  
        **kwargs
    ):
        self.model_id = model_id
        self.max_retries = max_retries
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.llm = self._load_model(**kwargs)
        
    def _load_model(self, **kwargs) -> LLM:
        print(f"Loading model: {self.model_id}...")
        torch.backends.cuda.matmul.allow_tf32 = True      
        torch.backends.cudnn.allow_tf32  = True

        return LLM(
            model=self.model_id,
            dtype="bfloat16", 
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.97, 
            max_model_len=8096, 
            max_num_batched_tokens=170000,  
            max_num_seqs=256, 
            **kwargs
        )
    
    def batch_infer_score(self, prompts: List[str]) -> List[dict]:
        sampling_params = SamplingParams(
            top_p=0.95,
            max_tokens=6300
        )
        
        original_prompts = list(prompts)
        results: List[dict] = [None] * len(original_prompts)
        remaining_idxs = list(range(len(original_prompts)))
        
        # Retry loop for failed extractions
        for attempt in range(1, self.max_retries + 1):
            if not remaining_idxs:
                break
                
            # Get prompts that still need processing
            batch_prompts = [original_prompts[i] for i in remaining_idxs]
            
            print(f"\nAttempt {attempt}/{self.max_retries} with {len(batch_prompts)} prompts")
            start_time = time.time()
            
            try:
                print(f"Starting inference...")
                outputs = self.llm.generate(batch_prompts, sampling_params)
                
                # Calculate timing stats
                elapsed = time.time() - start_time
                rate = len(batch_prompts) / elapsed if elapsed > 0 else 0
                print(f"Inference complete | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} prompts/s")
                
                # Extract texts from outputs
                texts = [out.outputs[0].text.strip() for out in outputs]
                
                # Process results and identify failed extractions
                next_remaining = []
                for idx, text in zip(remaining_idxs, texts):
                    matches = re.findall(
                        r"Score A:\s*(\d)\s*Score B:\s*(\d)\s*Score C:\s*([01](?:\.\d+)?)",
                        text,
                        re.DOTALL
                    )
                    if matches:
                        last = matches[-1]
                        results[idx] = {
                            "score_A": int(last[0]),
                            "score_B": int(last[1]),
                            "score_C": float(last[2]),
                            "score_O": float(last[2]),  # Initialize with same as C
                        }
                    else:
                        next_remaining.append(idx)
                
                remaining_idxs = next_remaining
                
                # Report progress
                successful = len(batch_prompts) - len(remaining_idxs)
                print(f"Successfully extracted scores from {successful}/{len(batch_prompts)} prompts")
                
                if remaining_idxs:
                    print(f"{len(remaining_idxs)} prompts need retry")
                    
            except Exception as e:
                print(f"Error during inference: {e}")
                if attempt < self.max_retries:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Max retries reached. Assigning fallback scores.")
                    break
        
        # Assign fallback scores for any still failed
        for idx in remaining_idxs:
            results[idx] = {"score_A": -1, "score_B": -1, "score_C": -1, "score_O": -1}
        
        if remaining_idxs:
            print(f"Assigned fallback scores to {len(remaining_idxs)} prompts")
        
        return results

# Global model instance
model_wrapper = None

def run_parallel_inference(prompts, indices=None):
    # initialize model wrapper class
    global model_wrapper
    if model_wrapper is None:
        print("Initializing model wrapper...")
        model_wrapper = VLLMInferenceWithRetry(
            tensor_parallel_size=8,
            trust_remote_code=True,
            batch_size=BATCH_SIZE
        )

    # log total prompts to process and start time
    print(f"\nTotal prompts to process: {len(prompts)}")
    start_time = time.time()

    # get results
    results = model_wrapper.batch_infer_score(prompts)

    # compute time elapsed
    elapsed = time.time() - start_time
    rate = len(prompts) / elapsed
    print(f"All prompts complete | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} prompts/s")

    # return results
    return results


def main():
    # A few torch optimizations
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        torch._dynamo.config.suppress_errors = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)

    # Open dataset
    dataset_path = Path("./dataset/processed_mistral_dataset.json")
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        full_records = json.load(f)

    # Separate records by mistral_score for monitoring purposes
    score_1_records = []
    score_0_records = []
    other_records = []  # Records without mistral_score or with unexpected values

    print("Categorizing records by mistral score...")
    for i, record in tqdm(enumerate(full_records), total=len(full_records), desc="Categorizing"):
        mistral_score = record.get("mistral_score")
        if mistral_score == 1:
            score_1_records.append((i, record))
        elif mistral_score == 0:
            score_0_records.append((i, record))
        else:
            other_records.append((i, record))
    
    print(f"\nRecords with mistral_score 1: {len(score_1_records)}")
    print(f"Records with mistral_score 0: {len(score_0_records)}")
    print(f"Records with other/no mistral_score: {len(other_records)}")

    # Sample 150K from each category
    import random
    random.seed(42)  # for reproducibility
    
    max_samples_per_category = 150000
    
    if len(score_1_records) > max_samples_per_category:
        print(f"Randomly sampling {max_samples_per_category} records from {len(score_1_records)} with mistral_score=1")
        score_1_records = random.sample(score_1_records, max_samples_per_category)
    
    if len(score_0_records) > max_samples_per_category:
        print(f"Randomly sampling {max_samples_per_category} records from {len(score_0_records)} with mistral_score=0")
        score_0_records = random.sample(score_0_records, max_samples_per_category)
    
    # Create a new records list with only the sampled records
    sampled_indices = [idx for idx, _ in score_1_records + score_0_records]
    records = [None] * len(full_records)  # Initialize with None for all indices
    
    # Fill in only the sampled records
    for idx, record in score_1_records + score_0_records:
        records[idx] = record
    
    # Discard other records (no mistral_score or unexpected values)
    discarded_indices = set(range(len(full_records))) - set(sampled_indices)
    
    print(f"\nSelected {len(score_1_records)} records with mistral_score=1 and {len(score_0_records)} records with mistral_score=0")
    print(f"Total sampled records: {len(score_1_records) + len(score_0_records)}")
    
    # Prepare for inference on sampled valid records
    valid_records = score_1_records + score_0_records
    valid_indices = [idx for idx, _ in valid_records]
    
    # Sort by index to maintain original order
    valid_records.sort(key=lambda x: x[0])
    
    # Prepare prompts for all valid records
    print("\nPreparing prompts for all valid records...")
    raw_prompts = [CLS_PROMPT.format(context=r["context"]) for _, r in valid_records]
    
    print("\nApplying chat template to prompts...")
    prompts = []
    for p in tqdm(raw_prompts, desc="Applying chat template"):
        prompts.append(
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True
            )
        )
    
    print(f"\nRunning inference on {len(prompts)} valid prompts...")
    overall_start = time.time()
    outputs = run_parallel_inference(prompts)
    
    # Check for disagreements and extraction failures
    failed_extraction = []
    disagreements = []
    
    print("\nProcessing inference results and checking for disagreements...")
    for i, ((idx, record), score) in enumerate(zip(valid_records, outputs)):
        mistral_score = record.get("mistral_score")
        
        # Check for score extraction failures
        if score["score_A"] == -1 and score["score_B"] == -1 and score["score_C"] == -1:
            failed_extraction.append((i, idx))
            continue
            
        # Check for disagreements based on the criteria
        if mistral_score == 1 and (score["score_A"] == 0 or score["score_B"] == 0 or score["score_C"] == 0):
            # If mistral_score is 1 but any model score is 0, mark as disagreement
            disagreements.append((i, idx))
            # Assign fallback scores (-1) for disagreements
            records[idx].update({
                "score_A": -1,
                "score_B": -1,
                "score_C": -1,
                "score_O": -1
            })
        elif mistral_score == 0 and (score["score_A"] == 1 and score["score_B"] == 1 and score["score_C"] == 1):
            # If mistral_score is 0 but all model scores are 1, mark as disagreement
            disagreements.append((i, idx))
            # Assign fallback scores (-1) for disagreements
            records[idx].update({
                "score_A": -1,
                "score_B": -1,
                "score_C": -1,
                "score_O": -1
            })
        else:
            # No disagreement, update record with scores
            records[idx].update(score)
    
    print(f"\nFound {len(disagreements)} disagreements between model and mistral scores.")
    print(f"Found {len(failed_extraction)} score extraction failures.")
    
    # Retry only for extraction failures, not disagreements
    retry_count = 0
    max_extraction_retries = 2
    
    while failed_extraction and retry_count < max_extraction_retries:
        retry_count += 1
        retry_indices = [i for i, _ in failed_extraction]
        retry_record_indices = [idx for _, idx in failed_extraction]
        
        retry_prompts = [prompts[i] for i in retry_indices]
        
        print(f"\nRetry {retry_count}/{max_extraction_retries}: Running inference on {len(retry_prompts)} extraction failure prompts...")
        retry_outputs = run_parallel_inference(retry_prompts)
        
        # Update extraction failures list for next retry
        new_failed_extraction = []
        for i, ((retry_idx, record_idx), score) in enumerate(zip(failed_extraction, retry_outputs)):
            mistral_score = records[record_idx].get("mistral_score")
            
            # Check if extraction still failed
            if score["score_A"] == -1 and score["score_B"] == -1 and score["score_C"] == -1:
                if retry_count < max_extraction_retries:
                    new_failed_extraction.append((retry_idx, record_idx))
                continue
                
            # Check for disagreements
            if mistral_score == 1 and (score["score_A"] == 0 or score["score_B"] == 0 or score["score_C"] == 0):
                # If mistral_score is 1 but any model score is 0, mark as disagreement
                disagreements.append((retry_idx, record_idx))
                # Assign fallback scores (-1) for disagreements
                records[record_idx].update({
                    "score_A": -1,
                    "score_B": -1,
                    "score_C": -1,
                    "score_O": -1
                })
            elif mistral_score == 0 and (score["score_A"] == 1 and score["score_B"] == 1 and score["score_C"] == 1):
                # If mistral_score is 0 but all model scores are 1, mark as disagreement
                disagreements.append((retry_idx, record_idx))
                # Assign fallback scores (-1) for disagreements
                records[record_idx].update({
                    "score_A": -1,
                    "score_B": -1,
                    "score_C": -1,
                    "score_O": -1
                })
            else:
                # No disagreement, update record with scores
                records[record_idx].update(score)
        
        failed_extraction = new_failed_extraction
    
    # After max retries, assign fallback scores to remaining extraction failures
    for _, record_idx in failed_extraction:
        records[record_idx].update({
            "score_A": -1,
            "score_B": -1,
            "score_C": -1,
            "score_O": -1
        })
    
    print(f"\nAssigned fallback scores to {len(failed_extraction)} records with persistent extraction failures.")
    
    # Overall timing stats
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal inference time: {overall_elapsed:.1f}s")
    print(f"Average time per prompt: {overall_elapsed/len(prompts):.2f}s")
    
    # Create final output dataset (only include sampled records)
    print("\nCreating final output dataset...")
    final_records = [record for record in records if record is not None]

    out_path = Path("./dataset/processed_mistral_dataset_scored.json")
    print(f"Saving final dataset to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Updated dataset saved to {out_path}")
    print(f"✓ Total records in final output: {len(final_records)}")


if __name__ == "__main__":
    main()