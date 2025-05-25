import time
import torch 
from typing import List
import json
from pathlib import Path
import re
import os
from transformers import AutoTokenizer
from prompt import CLS_PROMPT
from vllm import LLM, SamplingParams
from tqdm import tqdm

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
                
                # Log sample output for debugging
                if texts:
                    print(f"Sample output: {texts[0][:200]}...")
                
                # Save outputs for debugging
                with open("outputs.txt", "a", encoding="utf-8") as f:
                    for count, response in enumerate(texts):
                        f.write(f"=========Sample {count}=============\n")
                        f.write(response + "\n")
                        f.write(f"=========Sample {count}==============\n\n")
                
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
    dataset_path = Path("./dataset/processed_dataset_final.json")
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # if needed, modify the below line to process a subset of records
    records = records

    # Separate records by human_score for different processing
    score_1_records = []
    score_0_records = []
    score_neg1_records = []
    other_records = []  # Records without human_score or with unexpected values

    # Split records by human_score with progress bar
    print("Categorizing records by human score...")
    for i, record in tqdm(enumerate(records), total=len(records), desc="Categorizing"):
        human_score = record.get("human_score")
        if human_score == 1:
            score_1_records.append((i, record))
        elif human_score == 0:
            score_0_records.append((i, record))
        elif human_score == -1:
            score_neg1_records.append((i, record))
        else:
            other_records.append((i, record))
    
    print(f"\nRecords with human_score 1: {len(score_1_records)}")
    print(f"Records with human_score 0: {len(score_0_records)}")
    print(f"Records with human_score -1: {len(score_neg1_records)}")
    print(f"Records with other/no human_score: {len(other_records)}")

    # Process human_score 1 records: Skip inference, assign all 1's
    print("\nProcessing human_score 1 records...")
    for idx, record in tqdm(score_1_records, desc="Score 1 records"):
        records[idx].update({
            "score_A": 1,
            "score_B": 1,
            "score_C": 1.0,
            "score_O": 1.0
        })

    # Process human_score 0 records: Skip inference, assign A=B=1, C=O=0
    print("\nProcessing human_score 0 records...")
    for idx, record in tqdm(score_0_records, desc="Score 0 records"):
        records[idx].update({
            "score_A": 1,
            "score_B": 1,
            "score_C": 0.0,
            "score_O": 0.0
        })

    
    discarded_indices = set()
    
    # Discard other records (no human_score or unexpected values) - NO INFERENCE
    for idx, _ in other_records:
        discarded_indices.add(idx)
    
    print(f"\nAutomatically discarding {len(other_records)} records with missing/unexpected human labels.")
    
    if score_neg1_records:
        # Prepare for first inference round
        indices_neg1 = [idx for idx, _ in score_neg1_records]
        raw_prompts_neg1 = [CLS_PROMPT.format(context=r["context"]) for _, r in score_neg1_records]
        
        print("\nApplying chat template to prompts...")
        prompts_neg1 = []
        for p in tqdm(raw_prompts_neg1, desc="Applying chat template"):
            prompts_neg1.append(
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True
                )
            )
        
        print(f"\nRunning inference on {len(prompts_neg1)} prompts with human_score -1...")
        overall_start = time.time()
        outputs_neg1 = run_parallel_inference(prompts_neg1)
        
        # Check for complete disagreements and handle them
        disagreements_indices = []
        for i, ((idx, _), score) in enumerate(zip(score_neg1_records, outputs_neg1)):
            # Set score_O to 0 for all human_score -1 records
            score["score_O"] = 0
            
            # Check if this is a complete disagreement (all scores are 1)
            if score["score_A"] == 1 and score["score_B"] == 1 and score["score_C"] == 1:
                disagreements_indices.append((i, idx))
            else:
                # No disagreement, update record with scores
                records[idx].update(score)
        
        # Handle disagreements with retries
        retry_count = 0
        max_disagreement_retries = 2
        
        while disagreements_indices and retry_count < max_disagreement_retries:
            retry_count += 1
            retry_indices = [i for i, _ in disagreements_indices]
            retry_record_indices = [idx for _, idx in disagreements_indices]
            
            retry_prompts = [prompts_neg1[i] for i in retry_indices]
            
            print(f"\nRetry {retry_count}/{max_disagreement_retries}: Running inference on {len(retry_prompts)} disagreement prompts...")
            retry_outputs = run_parallel_inference(retry_prompts)
            
            # Update disagreements list for next retry
            new_disagreements = []
            for i, ((retry_idx, record_idx), score) in enumerate(zip(disagreements_indices, retry_outputs)):
                # Set score_O to 0 for all human_score -1 records
                score["score_O"] = 0
                
                # Check if still a complete disagreement
                if score["score_A"] == 1 and score["score_B"] == 1 and score["score_C"] == 1:
                    if retry_count < max_disagreement_retries:
                        new_disagreements.append((retry_idx, record_idx))
                else:
                    # No longer a disagreement, update record
                    records[record_idx].update(score)
            
            disagreements_indices = new_disagreements
        
        # After max retries, discard remaining disagreements
        for _, record_idx in disagreements_indices:
            discarded_indices.add(record_idx)
        
        print(f"\nDiscarding {len(disagreements_indices)} records due to persistent disagreements.")
        
        # Overall timing stats
        overall_elapsed = time.time() - overall_start
        print(f"\nTotal inference time: {overall_elapsed:.1f}s")
        print(f"Average time per prompt: {overall_elapsed/len(prompts_neg1):.2f}s")
    
    # Create final output dataset excluding discarded records
    print("\nCreating final output dataset...")
    final_records = [record for i, record in tqdm(enumerate(records), total=len(records), desc="Filtering") 
                     if i not in discarded_indices]

    out_path = Path("./dataset/Disc_PRM_ds.json")
    print(f"Saving final dataset to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Updated dataset saved to {out_path}")
    print(f"✓ Total records in final output: {len(final_records)}")


if __name__ == "__main__":
    main()